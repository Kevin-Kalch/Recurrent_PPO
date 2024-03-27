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
            nn.LeakyReLU(),
            self.init_weights(nn.Linear(64, 128)),
            nn.LeakyReLU(),
        )
        self.core = nn.Sequential(
            self.init_weights(nn.Linear(128+16, 128)), # 128 RAM
            nn.LeakyReLU(),
        )
        self.rnn = nn.Sequential(
            nn.Linear(128+16, 16),
            nn.Tanh(),
            nn.LayerNorm(16)
        )

        self.policy = nn.Sequential(
            self.init_weights(nn.Linear(128, 64), std=0.01),
            nn.LeakyReLU(),
            self.init_weights(nn.Linear(64, num_actions), std=0.01),
            nn.Softmax(dim=1)
        )

        self.value = nn.Sequential(
            self.init_weights(nn.Linear(128, 64), std=1),
            nn.LeakyReLU(),
            self.init_weights(nn.Linear(64, 1), std=1),
        )
        #self.popart_value = PopArtLayer(64, 1)

    def forward(self, obs, h):
        if h is None:
            h = torch.zeros((obs.size(0), 16))
        body_out = self.body(obs)
        core_out = self.core(torch.concat((body_out, h), dim=1))
        h_new = self.rnn(torch.concat((core_out, h), dim=1))
        return self.policy(core_out), self.value(core_out), h_new
    
    def get_action(self, obs, h):
        if h is None:
            h = torch.zeros((obs.size(0), 16))
        body_out = self.body(obs)
        core_out = self.core(torch.concat((body_out, h), dim=1))
        h_new = self.rnn(torch.concat((core_out, h), dim=1))

        policy_out = self.policy(core_out)
        return policy_out, h_new
    
    def get_value(self, obs, h):
        if h is None:
            h = torch.zeros((obs.size(0), 16))
        body_out = self.body(obs)
        core_out = self.core(torch.concat((body_out, h), dim=1))
        h_new = self.rnn(torch.concat((core_out, h), dim=1))
        value_out = self.value(core_out)
        return value_out, h_new
    
    def init_weights(self, layer, std=np.sqrt(2), bias=0.0):
        nn.init.orthogonal_(layer.weight, std)
        layer.bias.data.fill_(bias)
        return layer

# Memory als Liste von Tupels oder als mehrere Listen?
# Entscheidung: Mehrere Listen, weil im Tuple unklar ist welche Position welche variable ist.
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
    def __init__(self, num_in, num_actions, env_buffer_size, gamma=0.99):
        super(PPO, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.orig_model = ACNetwork(num_in, num_actions)
        self.model = self.orig_model # torch.compile(self.orig_model, mode="reduce-overhead")
        #self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, eps=1e-5)
        self.model.to(self.device)
        self.optimizer = torch.optim.RAdam([
            {'params': self.model.body.parameters(), 'lr': 3e-4},
            {'params': self.model.rnn.parameters(), 'lr': 3e-4},
            {'params': self.model.core.parameters(), 'lr': 3e-4},
            {'params': self.model.value.parameters(), 'lr': 3e-4},
            {'params': self.model.policy.parameters(), 'lr': 3e-4},
            ])
        self.gamma = gamma
        self.eps_clip = 0.2
        
        self.memory = {}
        
        self.env_buffer_size = env_buffer_size
        self.state_size = num_in
        self.obs_max = nn.Parameter(torch.ones((num_in), dtype=torch.float32), requires_grad=False)
        self.obs_max[:] = 1
        self.target_kl = 0.05
        self.beta = 3
        self.num_actions = num_actions
        self.burn_in_steps = 4

    @torch.no_grad()
    def select_action(self, state, h):
        state = torch.from_numpy(state).clone().float().unsqueeze(0)
        state = state / self.obs_max

        self.model = self.model.cpu()
        #state = state.to(self.device)

        probs, h_new = self.model.get_action(state, h)

        probs = probs[0].cpu()
        #m = Categorical(probs, validate_args=False)
        #action = m.sample() # to slow
        #log_prob = m.log_prob(action)
        action = (probs.cumsum(-1) >= rand(probs.shape[:-1])[..., None]).byte().argmax(-1)
        return action.detach().item(), probs.detach(), h_new.detach()
    
    def record_obs(self, state, hidden_state, action, reward, next_state, terminated, truncated, probs, env_id, step):
        # Define and if needed recalculate obs_max for all states
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)

        old_obs_max = self.obs_max.clone()
        self.obs_max.data = torch.max(torch.stack((self.obs_max, torch.abs(state))), axis=0)[0]
        self.obs_max.data = self.obs_max.data.type(torch.float32)
        if any(self.obs_max != old_obs_max):
            for id in self.memory.keys():
                self.memory[id].state = self.memory[id].state * old_obs_max / self.obs_max
                self.memory[id].next_state = self.memory[id].next_state * old_obs_max / self.obs_max

        state = state / self.obs_max
        next_state = next_state / self.obs_max

        if not env_id in self.memory:
            self.memory[env_id] = Memory(self.env_buffer_size, self.state_size, self.num_actions)
        self.memory[env_id].append(step, state, hidden_state, action, reward, next_state, terminated, truncated, probs)

    def train_epochs_bptt(self):
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
        batch_seq_length = 8
        
        num_samples = len(states.flatten(0,1))
        idx = np.arange(num_samples)
        done_split = np.where(dones.flatten(0,1) == True)[0] + 1
        env_split = np.where(idx%128 == 0)[0]
        splits = np.append(done_split, env_split)
        splits.sort()
        episodes = np.split(idx, splits)

        sequences = []
        for ep in episodes:
            if len(ep) < batch_seq_length: continue
            chunks1 = np.split(ep, np.arange(batch_seq_length,len(ep),batch_seq_length))
            if len(chunks1[-1]) != batch_seq_length:
                chunks1[-1] = ep[-batch_seq_length:]

            chunks2 = np.split(ep, np.arange(batch_seq_length+(batch_seq_length//2),len(ep),batch_seq_length))
            if len(chunks2[0]) != batch_seq_length:
                chunks2[0] = ep[(batch_seq_length//2):(batch_seq_length//2)+batch_seq_length]
            if len(chunks2[-1]) != batch_seq_length:
                chunks2[-1] = ep[-batch_seq_length-(batch_seq_length//2):-(batch_seq_length//2)]
    	    
            if all([len(c)==batch_seq_length for c in chunks2]):
                chunks = np.concatenate((chunks1, chunks2))
            else:
                chunks = chunks1
            #chunks = np.array([np.arange(idx, batch_seq_length+idx) for idx in ep[:-batch_seq_length]])
            sequences.extend(chunks)
        sequences = np.array(sequences)

        num_samples = len(sequences)
        idx = np.arange(num_samples)
        self.num_minibatches = 16
        splits = np.linspace(
            0, num_samples, self.num_minibatches+1, dtype=int)[1:-1]
        rnd.shuffle(idx)

        policy_losses = []
        value_losses = []
        ents = []

        #advantages = advantages.to(self.device)
        #returns = returns.to(self.device)
        #values = values.to(self.device)
        dones = dones.to(self.device)
        
        continue_training = True
        norms = []
        for epoch in range(10):
            h = hidden_states[:, 0, :]
            values = []
            next_values = []
            kl_divs = []
            with torch.no_grad():
                for i in range(128): # seq length, :]
                    probs, value, h = self.model(states[:, i, :], h)
                    next_value, _ = self.model.get_value(next_states[:, i, :], h)
                    
                    values.append(value.squeeze())
                    next_values.append(next_value.squeeze())

                    if i != (128 - 1):
                        done_mask = (dones[:, i] == True).to(self.device)
                        not_done_mask = (dones[:, i] == False).to(self.device)
                        # h = (h * not_done_mask[:, None]) + (hidden_states[:, i+1, :] * done_mask[:, None])
                        # hidden_states[:, i+1, :] = h 
                        h[done_mask] = hidden_states[done_mask, i+1, :]
                        hidden_states[not_done_mask, i+1, :] = h[not_done_mask] 
                    
                    # kl_div = torch.nn.functional.kl_div(torch.log(probs), old_probs[:, i, :], reduction="batchmean") # Targets should be probs, input in log space
                    # kl_divs.append(kl_div)
            # if epoch == 5:
            #     kl_divs = torch.stack(kl_divs).cpu().mean()
            #     if kl_divs < self.target_kl/1.5:
            #         self.beta = self.beta / 2
            #     if kl_divs > self.target_kl*1.5:
            #         self.beta = self.beta * 2

            #     print(str(self.beta) + " " + str(kl_divs.item()))

            if epoch == 9 or continue_training == False:
                break
                

            values = torch.stack(values, dim=1).cpu()
            next_values = torch.stack(next_values, dim=1).cpu()
            advantages, returns = self.calculate_advantages_returns(rewards, values, next_values, terminated, truncated, self.gamma, 0.95, True)
            
            ep_states = states.flatten(0,1)
            ep_hidden_states = hidden_states.flatten(0,1)
            ep_actions = actions.flatten(0,1)
            ep_old_probs = old_probs.flatten(0,1)
            ep_advantages = advantages.flatten(0,1).to(self.device)
            ep_returns = returns.flatten(0,1).to(self.device)
            #ep_values = values.flatten(0,1)
            ep_dones = dones.flatten(0,1)
            continue_training = True

            rnd.shuffle(idx)

            for k in np.split(idx, splits):
                states_idx = sequences[k] 

                seq_probs = torch.zeros((states_idx.shape[0], batch_seq_length, self.num_actions), dtype=torch.float32, device=self.device)
                seq_pred_values = torch.zeros((states_idx.shape[0], batch_seq_length), dtype=torch.float32, device=self.device)

                h = ep_hidden_states[states_idx[:, 0]]
                hs = [h]
                for i in range(batch_seq_length):

                    probs, pred_value, h = self.model(ep_states[states_idx[:, i]], h)
                    pred_value = pred_value.squeeze()
                    seq_probs[:, i] = probs
                    seq_pred_values[:, i] = pred_value
                    
                    # for idxs in range(states_idx.shape[0]):
                    #     h_idx = states_idx[idxs, i]+1
                    #     if h_idx < ep_hidden_states.shape[0] and h_idx % 128 != 0:
                    #         ep_hidden_states[h_idx] = h[idxs].clone().detach()
                    
                    hs.append(h)
                # m = Categorical(seq_probs.flatten(0,1), validate_args=False)
                # ent = m.entropy()
                #log_probs_o = m.log_prob(ep_actions[states_idx])
                # t = torch.gather(seq_probs, 0, ep_actions[states_idx][..., None])
                # t2 = torch.gather(seq_probs, 1, ep_actions[states_idx][..., None])
                t3 = torch.gather(seq_probs, -1, ep_actions[states_idx][..., None])
                log_probs = torch.log(t3)
                ent = -(seq_probs * log_probs).sum(dim=-1)
                log_probs = log_probs.squeeze()

                b_old_probs = ep_old_probs[states_idx]
                b_old_log_probs = torch.log(torch.gather(b_old_probs, -1, ep_actions[states_idx][..., None])).squeeze()

                ratios = torch.exp(log_probs - b_old_log_probs)
                surr1 = ratios * ep_advantages[states_idx]
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * ep_advantages[states_idx]
                
                policy_loss = (-torch.min(surr1, surr2)).mean()
                #pred_value_clipped = ep_valures[states_idx] + torch.clamp(seq_pred_values - ep_values[states_idx], -0.2, 0.2)
                #value_loss = 0.5 * torch.max((seq_pred_values-ep_returns[states_idx]).pow(2), (pred_value_clipped-ep_returns[states_idx]).pow(2)).mean()
                value_loss = (seq_pred_values-ep_returns[states_idx]).pow(2).mean()
                entropy_loss = -torch.clamp(ent, max=1.).mean()

                loss = policy_loss + 0.5*value_loss# + 1e-5 * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                # Gradienten clippen
                torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                ents.append(ent.mean().item())

                with torch.no_grad():
                    log_ratio = log_probs - b_old_log_probs
                    log_ratio = log_ratio.flatten()
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    deviations = torch.mean(torch.abs(torch.exp(log_ratio) - 1)).cpu().numpy()

                    self.target_kl = 0.05
                    if self.target_kl is not None and abs(approx_kl_div) > 1.5 * self.target_kl:
                        continue_training = False
                        print(f"Early stopping due to reaching max kl: {approx_kl_div:.2f}")

                    self.target_dev = 0.25
                    if self.target_dev is not None and abs(deviations) > self.target_dev:
                        continue_training = False
                        print(f"Early stopping due to reaching max deviation: {deviations:.2f}")
                
                if continue_training == False:
                   break
            if continue_training == False:
               break
        print("Policy Loss: " + str(sum(policy_losses) / len(policy_losses)) + "Value Loss: " + str(sum(value_losses) / len(value_losses)) + "Ent Loss: " + str(sum(ents) / len(ents)))
        #self.memory = {}

    def train_epochs_bptt_2(self):
        self.model = self.model.to(self.device)

        states = torch.stack([traj.state for traj in self.memory.values()])
        hidden_states = torch.stack([traj.hidden_state for traj in self.memory.values()])
        old_probs = torch.stack([traj.log_prob for traj in self.memory.values()])
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
        batch_seq_length = 8
        
        num_samples = len(states.flatten(0,1))
        idx = np.arange(num_samples)
        done_split = np.where(dones.flatten(0,1) == True)[0] + 1
        env_split = np.where(idx%512 == 0)[0][1:-1]
        splits = np.append(done_split, env_split)
        splits.sort()
        episodes = np.split(idx, splits)

        sequences = []
        for ep in episodes:
            if len(ep) < batch_seq_length: continue
            chunks = np.split(ep, np.arange(batch_seq_length,len(ep),batch_seq_length))
            if len(chunks[-1]) != batch_seq_length:
                chunks[-1] = ep[-batch_seq_length:]
            sequences.extend(chunks)
        sequences = np.array(sequences)

        num_samples = len(sequences)
        idx = np.arange(num_samples)
        self.num_minibatches = 16
        splits = np.linspace(
            0, num_samples, self.num_minibatches+1, dtype=int)[1:-1]
        #rnd.shuffle(idx)

        policy_losses = []
        value_losses = []
        ents = []

        #advantages = advantages.to(self.device)
        #returns = returns.to(self.device)
        #values = values.to(self.device)
        dones = dones.to(self.device)

        h = hidden_states[:, 0, :]
        values = []
        next_values = []
        with torch.no_grad():
            for i in range(512): # seq length,
                value, h = self.model.get_value(states[:, i, :], hidden_states[:, i, :])
                next_value, _ = self.model.get_value(next_states[:, i, :], h)
                
                values.append(value.squeeze())
                next_values.append(next_value.squeeze())

                # if i != (128 - 1):
                #     done_mask = (dones[:, i] == True).type(torch.IntTensor).to(self.device)
                #     not_done_mask = (dones[:, i] == False).type(torch.IntTensor).to(self.device)
                #     h = (h * not_done_mask[:, None]) + (hidden_states[:, i+1, :] * done_mask[:, None])
                #     hidden_states[:, i+1, :] = h

        values = torch.stack(values, dim=1).cpu()
        next_values = torch.stack(next_values, dim=1).cpu()
        advantages, returns = self.calculate_advantages_returns(rewards, values, next_values, terminated, truncated, self.gamma, 0.95, True)
        
        for epoch in range(5):
            ep_states = states.flatten(0,1)
            ep_hidden_states = hidden_states.flatten(0,1)
            ep_actions = actions.flatten(0,1)
            ep_old_probs = old_probs.flatten(0,1)
            ep_advantages = advantages.flatten(0,1).to(self.device)
            ep_returns = returns.flatten(0,1).to(self.device)
            #ep_values = values.flatten(0,1)
            ep_dones = dones.flatten(0,1)
            continue_training = True
            #rnd.shuffle(idx)

            for k in np.split(idx, splits):
                states_idx = sequences[k] 
                #ep_advantages[states_idx] = (ep_advantages[states_idx] - ep_advantages[states_idx].mean()) / (ep_advantages[states_idx].std() + 1e-8)

                h = ep_hidden_states[states_idx[:, 0]]
                h_states = [(None, h)]
                for i in range(batch_seq_length):
                    in_h = h_states[-1][1].detach()
                    in_h.requires_grad=True
                    #in_h.retain_grad()
                    probs, pred_value, h = self.model(ep_states[states_idx[:, i]], in_h)
                    

                    pred_value = pred_value.squeeze()
                    # m = Categorical(seq_probs.flatten(0,1), validate_args=False)
                    # ent = m.entropy()
                    #log_probs = m.log_prob(ep_actions[states_idx[:, i]])
                    log_probs = torch.log(torch.gather(probs, 1, ep_actions[states_idx[:, i]][..., None]))
                    ent = -(probs * log_probs).sum(dim=-1)

                    ratios = torch.exp(log_probs.squeeze() - ep_old_probs[states_idx[:, i]])
                    surr1 = ratios * ep_advantages[states_idx[:, i]]
                    surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * ep_advantages[states_idx[:, i]]
                    
                    policy_loss = (-torch.min(surr1, surr2)).mean()
                    #pred_value_clipped = ep_values[states_idx[:, i]] + torch.clamp(seq_pred_values - ep_values[states_idx[:, i]], -0.2, 0.2)
                    #value_loss = 0.5 * torch.max((seq_pred_values-ep_returns[states_idx[:, i]]).pow(2), (pred_value_clipped-ep_returns[states_idx[:, i]]).pow(2)).mean()
                    value_loss = 0.5*(pred_value-ep_returns[states_idx[:, i]]).pow(2).mean()
                    entropy_loss = -torch.clamp(ent, max=1.).mean()

                    loss = policy_loss + 0.5*value_loss# + 1e-5 * entropy_loss

                    # for idxs in range(states_idx.shape[0]):
                    #     h_idx = states_idx[idxs, i]+1
                    #     if h_idx < ep_hidden_states.shape[0] and h_idx % 128 != 0:
                    #         ep_hidden_states[h_idx] = h[idxs].detach()

                    h_states.append((in_h, h))

                    if True:
                        # https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500/63
                        self.optimizer.zero_grad()
                        for j in range(1, i+2):
                            if h_states[j][0].grad is not None:
                                h_states[j][0].grad.zero_()

                        loss.backward(retain_graph=True)
                        for j in range(i+1):
                            if h_states[-j-2][0] is None:
                                break
                            curr_grad = h_states[-j-1][0].grad.clone()
                            h_states[-j-2][1].backward(curr_grad, retain_graph=True)

                    # Gradienten clippen
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 0.50)
                    self.optimizer.step()

                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    ents.append(ent.mean().item())

                    with torch.no_grad():
                        log_ratio = log_probs.squeeze() - ep_old_probs[states_idx[:, i]]
                        log_ratio = log_ratio.flatten()
                        approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        deviations = torch.mean(torch.abs(torch.exp(log_ratio) - 1)).cpu().numpy()

                        self.target_kl = 0.05
                        if self.target_kl is not None and abs(approx_kl_div) > 1.5 * self.target_kl:
                            continue_training = False
                            print(f"Early stopping due to reaching max kl: {approx_kl_div:.2f}")

                        self.target_dev = 0.25
                        if self.target_dev is not None and abs(deviations) > self.target_dev:
                            continue_training = False
                            print(f"Early stopping due to reaching max deviation: {deviations:.2f}")

                    if continue_training == False:
                        break
                if continue_training == False:
                   break
            if continue_training == False:
               break
        print("Policy Loss: " + str(sum(policy_losses) / len(policy_losses)) + "Value Loss: " + str(sum(value_losses) / len(value_losses)) + "Ent Loss: " + str(sum(ents) / len(ents)))
        #self.memory = {}

    def train_epochs_bptt_3(self):
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
        batch_seq_length = 16
        
        num_samples = len(states.flatten(0,1))
        idx = np.arange(num_samples)
        done_split = np.where(dones.flatten(0,1) == True)[0] + 1
        env_split = np.where(idx%128 == 0)[0]
        splits = np.append(done_split, env_split)
        splits.sort()
        episodes = np.split(idx, splits)

        sequences = []
        for ep in episodes:
            if len(ep) < batch_seq_length: continue
            # chunks1 = np.split(ep, np.arange(batch_seq_length,len(ep),batch_seq_length))
            # if len(chunks1[-1]) != batch_seq_length:
            #     chunks1[-1] = ep[-batch_seq_length:]

            # chunks2 = np.split(ep, np.arange(batch_seq_length+(batch_seq_length//2),len(ep),batch_seq_length))
            # if len(chunks2[0]) != batch_seq_length:
            #     chunks2[0] = ep[(batch_seq_length//2):(batch_seq_length//2)+batch_seq_length]
            # if len(chunks2[-1]) != batch_seq_length:
            #     chunks2[-1] = ep[-batch_seq_length-(batch_seq_length//2):-(batch_seq_length//2)]
    	    
            # if all([len(c)==batch_seq_length for c in chunks2]):
            #     chunks = np.concatenate((chunks1, chunks2))
            # else:
            #     chunks = chunks1
            chunks = np.array([np.arange(idx, batch_seq_length+idx) for idx in ep[:-batch_seq_length]])
            sequences.extend(chunks)
        sequences = np.array(sequences)

        num_samples = len(sequences)
        idx = np.arange(num_samples)
        self.num_minibatches = 8
        splits = np.linspace(
            0, num_samples, self.num_minibatches+1, dtype=int)[1:-1]
        rnd.shuffle(idx)

        policy_losses = []
        value_losses = []
        ents = []

        #advantages = advantages.to(self.device)
        #returns = returns.to(self.device)
        #values = values.to(self.device)
        dones = dones.to(self.device)
        
        continue_training = True
        norms = []
        for epoch in range(10):
            h = hidden_states[:, 0, :]
            values = []
            next_values = []
            kl_divs = []
            with torch.no_grad():
                for i in range(128): # seq length, :]
                    probs, value, h = self.model(states[:, i, :], hidden_states[:, i, :])
                    next_value, _ = self.model.get_value(next_states[:, i, :], h)
                    
                    values.append(value.squeeze())
                    next_values.append(next_value.squeeze())

                    if i != (128 - 1):
                        done_mask = (dones[:, i] == True).to(self.device)
                        not_done_mask = (dones[:, i] == False).to(self.device)
                        # h = (h * not_done_mask[:, None]) + (hidden_states[:, i+1, :] * done_mask[:, None])
                        # hidden_states[:, i+1, :] = h 
                        h[done_mask] = hidden_states[done_mask, i+1, :]
                        hidden_states[not_done_mask, i+1, :] = h[not_done_mask] 
                    
                    # kl_div = torch.nn.functional.kl_div(torch.log(probs), old_probs[:, i, :], reduction="batchmean") # Targets should be probs, input in log space
                    # kl_divs.append(kl_div)
            # if epoch == 5:
            #     kl_divs = torch.stack(kl_divs).cpu().mean()
            #     if kl_divs < self.target_kl/1.5:
            #         self.beta = self.beta / 2
            #     if kl_divs > self.target_kl*1.5:
            #         self.beta = self.beta * 2

            #     print(str(self.beta) + " " + str(kl_divs.item()))

            if epoch == 9 or continue_training == False:
                break
                

            values = torch.stack(values, dim=1).cpu()
            next_values = torch.stack(next_values, dim=1).cpu()
            advantages, returns = self.calculate_advantages_returns(rewards, values, next_values, terminated, truncated, self.gamma, 0.95, True)

            ep_states = states.flatten(0,1)
            ep_hidden_states = hidden_states.flatten(0,1)
            ep_actions = actions.flatten(0,1)
            ep_old_probs = old_probs.flatten(0,1)
            ep_advantages = advantages.flatten(0,1).to(self.device)
            ep_returns = returns.flatten(0,1).to(self.device)
            #ep_values = values.flatten(0,1)
            ep_dones = dones.flatten(0,1)
            rnd.shuffle(idx)
            for k in np.split(idx, splits):
                states_idx = sequences[k] 

                h = ep_hidden_states[states_idx[:, 0]]
                total_loss = 0
                seq_probs = torch.zeros((states_idx.shape[0], batch_seq_length, self.num_actions), dtype=torch.float32, device=self.device)

                for i in range(batch_seq_length):
                    probs, pred_value, h = self.model(ep_states[states_idx[:, i]], h)
                    seq_probs[:, i] = probs
                    pred_value = pred_value.squeeze()

                    if i < self.burn_in_steps:
                        h = h.detach()
                        continue
                    # m = Categorical(probs.flatten(0,0), validate_args=False)
                    # ent = m.entropy()
                    # log_probs = m.log_prob(ep_actions[states_idx[:, i]])

                    log_probs = torch.log(torch.gather(probs, -1, ep_actions[states_idx[:, i]][..., None])) 
                    ent = -(probs * torch.log(probs)).sum(dim=-1)
                    log_probs = log_probs.squeeze()

                    b_old_probs = ep_old_probs[states_idx[:, i]]
                    b_old_log_probs = torch.log(torch.gather(b_old_probs, -1, ep_actions[states_idx[:, i]][..., None])).squeeze()

                    ratios = torch.exp(log_probs - b_old_log_probs)
                    surr1 = ratios * ep_advantages[states_idx[:, i]]
                    surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * ep_advantages[states_idx[:, i]]

                    kl_div = torch.nn.functional.kl_div(torch.log(probs), b_old_probs, reduction="batchmean") # Targets should be probs, input in log space
                    
                    policy_loss = (torch.min(surr1, surr2)).mean()# - self.beta* kl_div


                    pred_value_clipped = ep_returns[states_idx[:, i]] + torch.clamp(pred_value - ep_returns[states_idx[:, i]], -1.0, 1.0)
                    value_loss = 0.5 * (pred_value_clipped-ep_returns[states_idx[:, i]]).pow(2).mean()
                    #value_loss = 0.5*(pred_value-ep_returns[states_idx[:, i]]).pow(2).mean()
                    entropy_loss = -torch.clamp(ent, max=1.).mean()

                    loss = -(policy_loss - 0.5*value_loss + 1e-3 * entropy_loss)
                    total_loss += loss / batch_seq_length
                    #loss = loss / batch_seq_length
                    #loss.backward(retain_graph=True)

                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    ents.append(ent.mean().item())

                    # Update hidden
                    #done_mask = (ep_dones[states_idx[:, i]] == True).to(self.device)
                    #not_done_mask = (ep_dones[states_idx[:, i]] == False).to(self.device)

                    #next_state_idx = states_idx[:, i]+1
                    #next_state_idx = next_state_idx * (next_state_idx < len(ep_hidden_states)) + states_idx[:, i] * (next_state_idx == len(ep_hidden_states))
                    #h = torch.tensor((h * not_done_mask.unsqueeze(1)) + (ep_hidden_states[next_state_idx] * done_mask.unsqueeze(1)))
                    #h.retain_grad()

                    #h[done_mask] = ep_hidden_states[next_state_idx][done_mask]
                    # for idxs in range(states_idx.shape[0]):
                    #     h_idx = states_idx[idxs, i]+1
                    #     if h_idx < ep_hidden_states.shape[0] and h_idx % 128 != 0 and ep_dones[h_idx-1] == False:
                    #         ep_hidden_states[h_idx] = h[idxs].clone().detach()


                with torch.no_grad():
                    seq_log_probs = torch.log(torch.gather(seq_probs, -1, ep_actions[states_idx][..., None])).squeeze()

                    b_old_probs = ep_old_probs[states_idx]
                    b_old_log_probs = torch.log(torch.gather(b_old_probs, -1, ep_actions[states_idx][..., None])).squeeze()

                    log_ratio = seq_log_probs[self.burn_in_steps:] - b_old_log_probs[self.burn_in_steps:]
                    log_ratio = log_ratio.flatten()
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    deviations = torch.mean(torch.abs(torch.exp(log_ratio) - 1)).cpu().numpy()

                    kl_div = torch.nn.functional.kl_div(torch.log(seq_probs[self.burn_in_steps:].flatten(0,1)), b_old_probs[self.burn_in_steps:].flatten(0,1), reduction="batchmean")
                    if self.target_kl is not None and abs(kl_div) > 1.5 * self.target_kl:
                        continue_training = False
                        print(f"Early stopping due to reaching max kl: {kl_div:.2f}, aprox kl: {approx_kl_div:.2f}")
                        break

                    # self.target_dev = 0.25
                    # if self.target_dev is not None and abs(deviations) > self.target_dev:
                    #     continue_training = False
                    #     print(f"Early stopping due to reaching max deviation: {deviations:.2f}")
                    #     break
                
                
                
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
                self.optimizer.zero_grad()
        print("Policy Loss: " + str(sum(policy_losses) / len(policy_losses)) + " Value Loss: " + str(sum(value_losses) / len(value_losses)) + " Ent Loss: " + str(sum(ents) / len(ents)))
        if len(norms) > 0:
            print(f"{np.min(norms)} {np.mean(norms)} {np.max(norms)}")

    
    def save_model(self, path):
        torch.save(self.state_dict(), path + ".pt")

    def load_model(self, path):
        self.load_state_dict(torch.load(path + ".pt"))
        #self.model = torch.compile(self.orig_model)

    def calculate_advantages_returns(self, rewards, values, next_values, terminated, truncated, discount_factor, trace_decay, normalize):
        returns = []
        
        for traj_rewards, traj_values, traj_nvalues, traj_term, traj_trunc in zip(rewards, values, next_values, terminated, truncated):
            if traj_term[-1] == False:
                R = traj_nvalues[-1]
            traj_returns = []
            for reward, value, nvalue, term, trunc in zip(reversed(traj_rewards), reversed(traj_values), reversed(traj_nvalues), reversed(traj_term), reversed(traj_trunc)):
                if trunc:
                    R = nvalue
                if term:
                    R = 0
                R = reward + R * discount_factor

                traj_returns.insert(0, R)
            returns.append(traj_returns)

        returns = torch.tensor(returns)
        advantages = returns - values
        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            #returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return advantages, returns


class PopArtLayer(torch.nn.Module):

    def __init__(self, input_features, output_features, beta=4e-4):
        self.beta = beta

        super(PopArtLayer, self).__init__()

        self.input_features = input_features
        self.output_features = output_features

        self.weight = torch.nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias = torch.nn.Parameter(torch.Tensor(output_features))

        self.register_buffer('mu', torch.zeros(output_features, requires_grad=False))
        self.register_buffer('sigma', torch.ones(output_features, requires_grad=False))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):

        normalized_output = torch.matmul(inputs, self.weight.t())
        normalized_output += self.bias.unsqueeze(0).expand_as(normalized_output)

        with torch.no_grad():
            output = normalized_output * self.sigma + self.mu

        return [output, normalized_output]

    def update_parameters(self, vs):
        
        oldmu = self.mu
        oldsigma = self.sigma

        vs = vs# * task
        n = len(vs)#task.sum()
        mu = vs.sum() / n
        nu = torch.sum(vs**2) / n
        sigma = torch.sqrt(nu - mu**2)
        sigma = torch.clamp(sigma, min=1e-4, max=1e+6)
        

        if torch.isnan(mu):
            mu = self.mu.item()
        if torch.isnan(sigma):
            sigma = self.sigma.item()

        self.mu = (1 - self.beta) * self.mu + self.beta * mu
        self.sigma = (1 - self.beta) * self.sigma + self.beta * sigma

        self.weight.data = (self.weight.t() * oldsigma / self.sigma).t()
        self.bias.data = (oldsigma * self.bias + oldmu - self.mu) / self.sigma
