from random import Random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
from torch.distributions import Categorical
import math
from torch import rand
import copy

class ACNetwork(nn.Module):
    def __init__(self, num_in, num_actions) -> None:
        super(ACNetwork, self).__init__()
        self.body = nn.Sequential(
            self.init_weights(nn.Linear(num_in, 64)), # 128 RAM
            nn.Tanh(),
            self.init_weights(nn.Linear(64, 128)),
            nn.Tanh(),
        )
        self.core = nn.Sequential(
            self.init_weights(nn.Linear(128+16, 128)), # 128 RAM
            nn.Tanh(),
        )
        self.rnn = nn.GRUCell(128, 16)

        self.policy = nn.Sequential(
            self.init_weights(nn.Linear(128, 64), std=0.01),
            nn.Tanh(),
            self.init_weights(nn.Linear(64, num_actions), std=0.01),
            nn.Softmax(dim=1)
        )

        self.value = nn.Sequential(
            self.init_weights(nn.Linear(128, 64), std=1),
            nn.Tanh(),
            self.init_weights(nn.Linear(64, 1), std=1),
        )

    def forward(self, obs, h):
        if h is None:
            h = torch.zeros((obs.size(0), 16)).to(obs.device)
        body_out = self.body(obs)
        core_out = self.core(torch.concat((body_out, h), dim=1))
        h_new = self.rnn(core_out, h)
        return self.policy(core_out), self.value(core_out), h_new
    
    def get_action(self, obs, h):
        if h is None:
            h = torch.zeros((obs.size(0), 16)).to(obs.device)
        body_out = self.body(obs)
        core_out = self.core(torch.concat((body_out, h), dim=1))
        h_new = self.rnn(core_out, h)

        policy_out = self.policy(core_out)
        return policy_out, h_new
    
    def get_value(self, obs, h):
        if h is None:
            h = torch.zeros((obs.size(0), 16)).to(obs.device)
        body_out = self.body(obs)
        core_out = self.core(torch.concat((body_out, h), dim=1))
        h_new = self.rnn(core_out, h)
        value_out = self.value(core_out)
        return value_out, h_new
    
    def init_weights(self, layer, std=np.sqrt(2), bias=0.0):
        nn.init.orthogonal_(layer.weight, std)
        layer.bias.data.fill_(bias)
        return layer

class Memory:
    def __init__(self, num_samples, state_size, num_actions):
        self.state = torch.zeros((num_samples, state_size), dtype=torch.float32)
        self.hidden_state = torch.zeros((num_samples, 16), dtype=torch.float32)
        self.action = torch.zeros(num_samples, dtype=torch.long)
        self.reward = torch.zeros(num_samples, dtype=torch.float32)
        self.next_state = torch.zeros((num_samples, state_size), dtype=torch.float32)
        self.terminated = torch.zeros(num_samples, dtype=torch.bool)
        self.truncated = torch.zeros(num_samples, dtype=torch.bool)
        self.probs = torch.zeros((num_samples, num_actions), dtype=torch.float32)

    def append(self, index, state, hidden_state, action, reward, next_state, terminated, truncated, probs):
        self.state[index] = state
        if hidden_state is not None:
            self.hidden_state[index] = hidden_state
        self.action[index] = action
        self.reward[index] = reward
        self.next_state[index] = next_state
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self.probs[index] = probs

class PPO(nn.Module):
    def __init__(self, num_in, num_actions, env_buffer_size, gamma=0.99, writer=None):
        super(PPO, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.orig_model = ACNetwork(num_in, num_actions)
        self.model = self.orig_model # torch.compile(self.orig_model, mode="reduce-overhead")
        #self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, eps=1e-5)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4, fused=True, eps=1e-5)
        self.gamma = gamma
        self.eps_clip = 0.2
        
        self.memory = {}
        
        self.env_buffer_size = env_buffer_size
        self.state_size = num_in
        self.obs_max = nn.Parameter(torch.ones((num_in), dtype=torch.float32), requires_grad=False)
        #self.obs_max.data = torch.tensor([1.5, 1.5, 3.1415927, 1.0, 1.0], dtype=torch.float32)
        self.target_kl = 0.01 #0.01
        self.beta = 3
        self.num_actions = num_actions
        self.burn_in_steps = 2
        self.writer = writer

    @torch.no_grad()
    def select_action(self, state, h, eval=False):
        state = torch.from_numpy(state).clone().float().unsqueeze(0)
        #state = state / self.obs_max
        state = state.to(self.device)
        if h is not None:
            h = h.to(self.device)
        #self.model = self.model.cpu()
        probs, h_new = self.model.get_action(state, h)

        probs = probs[0].cpu()
        #action = (probs.cumsum(-1) >= rand(probs.shape[:-1])[..., None]).byte().argmax(-1)
        m = Categorical(probs)
        action = m.sample()
        if eval:
            action = probs.argmax()
        return action.detach().item(), probs.detach(), h_new.detach()
    
    def record_obs(self, state, hidden_state, action, reward, next_state, terminated, truncated, probs, env_id, step):
        # Define and if needed recalculate obs_max for all states
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)

        # old_obs_max = self.obs_max.clone()
        # self.obs_max.data = torch.max(torch.stack((self.obs_max, torch.abs(state))), axis=0)[0]
        # self.obs_max.data = self.obs_max.data.type(torch.float32)
        # if any(self.obs_max != old_obs_max):
        #     for id in self.memory.keys():
        #         self.memory[id].state = self.memory[id].state * old_obs_max / self.obs_max
        #         self.memory[id].next_state = self.memory[id].next_state * old_obs_max / self.obs_max

        #state = state / self.obs_max
        #next_state = next_state / self.obs_max

        if not env_id in self.memory:
            self.memory[env_id] = Memory(self.env_buffer_size, self.state_size, self.num_actions)
        self.memory[env_id].append(step, state, hidden_state, action, reward, next_state, terminated, truncated, probs)

    def train_epochs_bptt(self, optim_epoch):
        self.model = self.model.to(self.device)

        states = torch.stack([self.memory[key].state for key in sorted(self.memory.keys())])
        hidden_states = torch.stack([self.memory[key].hidden_state for key in sorted(self.memory.keys())])
        old_probs = torch.stack([self.memory[key].probs for key in sorted(self.memory.keys())])
        actions = torch.stack([self.memory[key].action for key in sorted(self.memory.keys())])
        next_states = torch.stack([self.memory[key].next_state for key in sorted(self.memory.keys())])
        rewards = torch.stack([self.memory[key].reward for key in sorted(self.memory.keys())])
        terminated = torch.stack([self.memory[key].terminated for key in sorted(self.memory.keys())])
        truncated = torch.stack([self.memory[key].truncated for key in sorted(self.memory.keys())])
        dones = torch.logical_or(terminated, truncated)

        states = states.to(self.device)
        hidden_states = hidden_states.to(self.device)
        old_probs = old_probs.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        #rewards = rewards.to(self.device)
        #dones = dones.to(self.device)

        rnd = Random()
        batch_seq_length = 8
        
        num_samples = len(states.flatten(0,1))
        idx = np.arange(num_samples)
        #done_split = np.where(dones.flatten(0,1) == True)[0] + 1
        env_split = np.where(idx%self.env_buffer_size == 0)[0]
        splits = env_split
        #splits = np.append(done_split, env_split)
        splits.sort()
        episodes = np.split(idx, splits)

        sequences = []
        for ep in episodes:
            if len(ep) < batch_seq_length: continue
            splits = np.linspace(0, len(ep), (len(ep)//8)+1, dtype=int)[1:-1]
            chunks = np.split(ep, splits)
            chunks[-1] = chunks[-1]-1
            sequences.extend(chunks)
        sequences = np.array(sequences)

        num_samples = len(sequences)
        idx = np.arange(num_samples)
        self.num_minibatches = 4
        splits = np.linspace(
            0, num_samples, self.num_minibatches+1, dtype=int)[1:-1]
        rnd.shuffle(idx)

        policy_losses = []
        value_losses = []
        ents = []
        kl_divs=[]
        norms = []
        kl_divs = []
        clip_fractions = []
        dones = dones.to(self.device)
        
        continue_training = True
        
        old_model = None
        for epoch in range(5):
            if epoch == 0 or True:
                values = []
                next_values = []
                with torch.no_grad():
                    for i in range(self.env_buffer_size): # seq length, :]
                        probs, value, h = self.model(states[:, i, :], hidden_states[:, i, :])
                        next_value, h_n = self.model.get_value(next_states[:, i, :], h)
                        
                        values.append(value.squeeze())
                        next_values.append(next_value.squeeze())

                        if i != (self.env_buffer_size-1):
                            done_mask = (dones[:, i] == True).to(self.device)
                            not_done_mask = (dones[:, i] == False).to(self.device)
                            h[done_mask] = hidden_states[done_mask, i+1, :]
                            hidden_states[:, i+1, :] = h

                values = torch.stack(values, dim=1).cpu()
                next_values = torch.stack(next_values, dim=1).cpu()
                advantages, returns = self.calculate_advantages_returns(rewards, values, next_values, terminated, truncated, self.gamma, 0.95, False)
                #advantages = torch.clamp(advantages, -0.5, 0.5)
            if epoch == 4 or continue_training == False:
                for i, key in enumerate(self.memory.keys()):
                    #self.memory[key].hidden_state = hidden_states[i].cpu()
                    pass
                break

            ep_states = states.flatten(0,1)
            ep_hidden_states = hidden_states.flatten(0,1)
            ep_actions = actions.flatten(0,1)
            ep_old_probs = old_probs.flatten(0,1)
            ep_advantages = advantages.flatten(0,1).to(self.device)
            ep_returns = returns.flatten(0,1).to(self.device)
            ep_dones = dones.flatten(0,1)
            rnd.shuffle(idx)
            for k in np.split(idx, splits):
                states_idx = sequences[k] 

                h = ep_hidden_states[states_idx[:, 0]]
                total_loss = 0
                seq_probs = torch.zeros((states_idx.shape[0], batch_seq_length, self.num_actions), dtype=torch.float32, device=self.device)
                for i in range(batch_seq_length):
                    #ep_advantages[states_idx[:, i]] = (ep_advantages[states_idx[:, i]] - ep_advantages[states_idx[:, i]].mean()) / (ep_advantages[states_idx[:, i]].std() + 1e-8)

                    probs, pred_value, h = self.model(ep_states[states_idx[:, i]], h)
                    seq_probs[:, i] = probs
                    pred_value = pred_value.squeeze()

                    log_probs = torch.log(torch.gather(probs, -1, ep_actions[states_idx[:, i]][..., None])) 
                    ent = -(probs * torch.log(probs)).sum(dim=-1)
                    log_probs = log_probs.squeeze()

                    b_old_probs = ep_old_probs[states_idx[:, i]]
                    b_old_log_probs = torch.log(torch.gather(b_old_probs, -1, ep_actions[states_idx[:, i]][..., None]))
                    b_old_log_probs = b_old_log_probs.squeeze()

                    ratios = torch.exp(log_probs - b_old_log_probs)
                    surr1 = ratios * ep_advantages[states_idx[:, i]]
                    surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * ep_advantages[states_idx[:, i]]
                    policy_loss = -(torch.min(surr1, surr2)).mean()
                    clip_fraction = torch.mean((torch.abs(ratios - 1) > self.eps_clip).float()).item()
                    clip_fractions.append(clip_fraction)

                    #pred_value_clipped = ep_returns[states_idx[:, i]] + torch.clamp(pred_value - ep_returns[states_idx[:, i]], -0.5, 0.5)
                    #value_loss = 0.5 * (ep_returns[states_idx[:, i]]-pred_value_clipped).pow(2).mean()
                    value_loss = 0.5*(ep_returns[states_idx[:, i]]-pred_value).pow(2).mean()
                    #value_loss = torch.max(value_loss_clipped, value_loss).mean()
                    entropy_loss = -torch.clamp(ent, max=1.).mean()

                    # Early stopping
                    with torch.no_grad():
                        log_ratio = log_probs - b_old_log_probs
                        approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        kl_divs.append(approx_kl_div)
                        if self.target_kl is not None and abs(approx_kl_div) > 1.5 * 0.03:
                            if old_model is not None:
                                self.model.load_state_dict(old_model)
                            continue_training = False
                            print(f"Early stopping due to reaching max kl: {approx_kl_div:.2f}, aprox kl: {approx_kl_div:.2f}")
                            break

                    #self.optimizer.zero_grad(set_to_none=True)
                    loss = policy_loss + 0.5 *value_loss + 0 * entropy_loss
                    total_loss += loss / batch_seq_length
                    #loss = loss / batch_seq_length
                    #loss.backward(retain_graph=True)
                    #torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 0.5)
                    #self.optimizer.step()

                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    ents.append(ent.mean().item())

                    done_mask = (ep_dones[states_idx[:, i]] == True).to(self.device).int()
                    not_done_mask = (ep_dones[states_idx[:, i]] == False).to(self.device).int()
                    h = h * not_done_mask[:, None] + ep_hidden_states[states_idx[:, i]+1] * done_mask[:, None]

                if continue_training == False:
                    break
                
                old_model = self.model.state_dict()
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                norms.append(total_norm)

                torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
        print("Policy Loss: " + str(sum(policy_losses) / len(policy_losses)) + " Value Loss: " + str(sum(value_losses) / len(value_losses)) + " Ent Loss: " + str(sum(ents) / len(ents)))
        if len(norms) > 0:
            print(f"{np.min(norms)} {np.mean(norms)} {np.max(norms)}")

        explained_var = explained_variance(values.flatten().cpu().numpy(), returns.flatten().cpu().numpy())
        if self.writer is not None:
            self.writer.add_scalar("train/policy_loss", sum(policy_losses) / len(policy_losses), optim_epoch)
            self.writer.add_scalar("train/value_loss", sum(value_losses) / len(value_losses), optim_epoch)
            self.writer.add_scalar("train/entropy_loss", sum(ents) / len(ents), optim_epoch)
            self.writer.add_scalar("train/norms_min", np.min(norms), optim_epoch)
            self.writer.add_scalar("train/norms_mean", np.mean(norms), optim_epoch)
            self.writer.add_scalar("train/norms_max", np.max(norms), optim_epoch)
            self.writer.add_scalar("train/KL_div", np.mean(kl_divs), optim_epoch)
            self.writer.add_scalar("train/explained_var", explained_var, optim_epoch)

    def train_epochs_bptt_2(self, optim_epoch):
        self.model = self.model.to(self.device)

        states = torch.stack([traj.state for traj in self.memory.values()])
        hidden_states = torch.stack([traj.hidden_state for traj in self.memory.values()])
        old_probs = torch.stack([traj.probs for traj in self.memory.values()])
        actions = torch.stack([traj.action for traj in self.memory.values()])
        next_states = torch.stack([traj.next_state for traj in self.memory.values()])
        rewards = torch.stack([traj.reward for traj in self.memory.values()])
        terminated = torch.stack([traj.terminated for traj in self.memory.values()])
        truncated = torch.stack([traj.truncated for traj in self.memory.values()])
        dones = torch.logical_or(terminated, truncated)

        states = states.to(self.device)
        hidden_states = hidden_states.to(self.device)
        old_probs = old_probs.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        #rewards = rewards.to(self.device)
        #dones = dones.to(self.device)

        rnd = Random()
        batch_seq_length = self.env_buffer_size//8

        policy_losses = []
        value_losses = []
        ents = []
        kl_divs=[]
        norms = []
        clip_fractions = []
        dones = dones.to(self.device)
        
        continue_training = True
        norms = []
        old_model = None
        for epoch in range(5):
            h = hidden_states[:, 0, :]
            with torch.no_grad():
                if epoch == 0 or True:
                    values = []
                    next_values = []
                    for i in range(self.env_buffer_size): # seq length, :]
                        probs, value, h = self.model(states[:, i, :], hidden_states[:, i, :])
                        next_value, h_n = self.model.get_value(next_states[:, i, :], h)
                        
                        values.append(value.squeeze())
                        next_values.append(next_value.squeeze())

                        if i != (self.env_buffer_size-1):
                            done_mask = (dones[:, i] == True).to(self.device)
                            not_done_mask = (dones[:, i] == False).to(self.device)
                            h[done_mask] = hidden_states[done_mask, i+1, :]
                            hidden_states[not_done_mask, i+1, :] = h[not_done_mask]
                    

                    values = torch.stack(values, dim=1).cpu()
                    next_values = torch.stack(next_values, dim=1).cpu()
                    advantages, returns = self.calculate_advantages_returns(rewards, values, next_values, terminated, truncated, self.gamma, 0.95, False)
                    advantages = advantages.to(self.device)
                    returns = returns.to(self.device)

                    if epoch == 4 or continue_training == False:
                        for i, key in enumerate(self.memory.keys()):
                            self.memory[key].hidden_state = hidden_states[i].cpu()
                        break

            h = hidden_states[:, 0, :]
            total_loss = 0
            seq_log_probs = torch.zeros((states.size(0), batch_seq_length), dtype=torch.float32, device=self.device)
            seq_old_log_probs = torch.zeros((states.size(0), batch_seq_length), dtype=torch.float32, device=self.device)
            for i in range(self.env_buffer_size): # seq length, :]
                probs, pred_value, h = self.model(states[:, i, :], h)
                pred_value = pred_value.squeeze()
                #advantages[:, i] = (advantages[:, i] - advantages[:, i].mean()) / (advantages[:, i].std() + 1e-8)

                log_probs = torch.log(torch.gather(probs, -1, actions[:, i][..., None])) 
                ent = -(probs * torch.log(probs)).sum(dim=-1)
                log_probs = log_probs.squeeze()
                seq_log_probs[:, i%batch_seq_length] = log_probs

                b_old_probs = old_probs[:, i, :]
                b_old_log_probs = torch.log(torch.gather(b_old_probs, -1, actions[:, i][..., None])).squeeze()
                seq_old_log_probs[:, i%batch_seq_length] = b_old_log_probs

                ratios = torch.exp(log_probs - b_old_log_probs)
                surr1 = ratios * advantages[:, i]
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages[:, i]
                clip_fraction = torch.mean((torch.abs(ratios - 1) > self.eps_clip).float()).item()
                clip_fractions.append(clip_fraction)
                
                policy_loss = -(torch.min(surr1, surr2)).mean()
                #pred_value_clipped = returns[:, i] + torch.clamp(pred_value - returns[:, i], -0.5, 0.5)
                #value_loss = 0.5 * (returns[:, i]-pred_value_clipped).pow(2).mean()
                value_loss = 0.5*(returns[:, i]-pred_value).pow(2).mean()
                entropy_loss = -torch.clamp(ent, max=1.).mean()

                loss = policy_loss + 1.0 * value_loss + 1e-5 * entropy_loss
                total_loss += loss / batch_seq_length
                #loss = loss / batch_seq_length
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                ents.append(ent.mean().item())

                done_mask = (dones[:, i] == True).to(self.device).int()
                not_done_mask = (dones[:, i] == False).to(self.device).int()
                if i != self.env_buffer_size-1:
                    h = h * not_done_mask[:, None] + hidden_states[:, i+1] * done_mask[:, None]
                    #hidden_states[not_done_mask, i+1, :] = h[not_done_mask].detach()
                
                if i%batch_seq_length == 0:
                    # Early stopping
                    with torch.no_grad():
                        log_ratio = seq_log_probs.flatten() - seq_old_log_probs.flatten()
                        approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        kl_divs.append(approx_kl_div)
                        if self.target_kl is not None and abs(approx_kl_div) > 1.5 * 0.03:
                            if old_model is not None:
                                self.model.load_state_dict(old_model)
                            continue_training = False
                            print(f"Early stopping due to reaching max kl: {approx_kl_div:.2f}, aprox kl: {approx_kl_div:.2f}")
                            break

                    old_model = self.model.state_dict()
                    self.optimizer.zero_grad()
                    total_loss.backward(retain_graph=True)
                    total_loss = 0

                    total_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    norms.append(total_norm)
                    #h = h.detach()

                    torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()
                
        print("Policy Loss: " + str(sum(policy_losses) / len(policy_losses)) + " Value Loss: " + str(sum(value_losses) / len(value_losses)) + " Ent Loss: " + str(sum(ents) / len(ents)))
        if len(norms) > 0:
            print(f"{np.min(norms)} {np.mean(norms)} {np.max(norms)}")
        
        explained_var = explained_variance(values.flatten().cpu().numpy(), returns.flatten().cpu().numpy())
        if self.writer is not None:
            self.writer.add_scalar("train/policy_loss", sum(policy_losses) / len(policy_losses), optim_epoch)
            self.writer.add_scalar("train/value_loss", sum(value_losses) / len(value_losses), optim_epoch)
            self.writer.add_scalar("train/entropy_loss", sum(ents) / len(ents), optim_epoch)
            self.writer.add_scalar("train/norms_min", np.min(norms), optim_epoch)
            self.writer.add_scalar("train/norms_mean", np.mean(norms), optim_epoch)
            self.writer.add_scalar("train/norms_max", np.max(norms), optim_epoch)
            self.writer.add_scalar("train/KL_div", np.mean(kl_divs), optim_epoch)
            self.writer.add_scalar("train/explained_var", explained_var, optim_epoch)
            self.writer.add_scalar("train/clip_fraction", np.mean(clip_fractions), optim_epoch)

    
    def save_model(self, path):
        torch.save(self.state_dict(), path + ".pt")

    def load_model(self, path):
        self.load_state_dict(torch.load(path + ".pt"))

    def calculate_advantages_returns(self, rewards, values, next_values, terminated, truncated, discount_factor, trace_decay, normalize):
        returns = []
        advantages = []
        for traj_rewards, traj_values, traj_nvalues, traj_term, traj_trunc in zip(rewards, values, next_values, terminated, truncated):
            adv = 0
            R = 0
            if traj_term[-1] == False:
                adv = traj_nvalues[-1] - traj_values[-1]
                R = traj_nvalues[-1]
            traj_returns = []
            tarj_advantages = []
            for reward, value, nvalue, term, trunc in zip(reversed(traj_rewards), reversed(traj_values), reversed(traj_nvalues), reversed(traj_term), reversed(traj_trunc)):
                if trunc:
                    R = 0
                    adv = 0
                    delta = reward - value
                elif term:
                    R = 0
                    adv = 0
                    delta = reward - value
                else:
                    delta = reward + discount_factor * nvalue - value
                adv = delta + discount_factor * trace_decay * adv
                R = reward + R * discount_factor

                traj_returns.insert(0, R)
                tarj_advantages.insert(0, adv)
            returns.append(traj_returns)
            advantages.append(tarj_advantages)

        returns = torch.tensor(returns)
        advantages = returns - values
        #advantages = torch.tensor(advantages)
        #returns = advantages + values
        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns
    
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y