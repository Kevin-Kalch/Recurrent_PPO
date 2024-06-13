from random import Random
import numpy as np
import torch
import torch.nn as nn
import math
from torch import rand

# https://github.com/Kaixhin/NoisyNet-A3C/blob/master/model.py
class NoisyLinear(nn.Linear):
  def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
    super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
    # µ^w and µ^b reuse self.weight and self.bias
    self.sigma_init = sigma_init
    self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features))  # σ^w
    self.sigma_bias = nn.Parameter(torch.Tensor(out_features))  # σ^b
    self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
    self.register_buffer('epsilon_bias', torch.zeros(out_features))
    self.reset_parameters()

  def reset_parameters(self):
    if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
      nn.init.uniform_(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      nn.init.uniform_(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      nn.init.constant_(self.sigma_weight, self.sigma_init)
      nn.init.constant_(self.sigma_bias, self.sigma_init)

  def forward(self, input):
    return torch.nn.functional.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, self.bias + self.sigma_bias * self.epsilon_bias)

  def sample_noise(self):
    self.epsilon_weight = torch.randn(self.out_features, self.in_features)
    self.epsilon_bias = torch.randn(self.out_features)

  def remove_noise(self):
    self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
    self.epsilon_bias = torch.zeros(self.out_features)

# https://arxiv.org/pdf/1706.01905
class ACNetwork(nn.Module):
    def __init__(self, num_in, num_actions) -> None:
        super(ACNetwork, self).__init__()
        self.num_in = num_in
        self.num_actions = num_actions

        if isinstance(num_in, int): # Linear input
            self.body = nn.Sequential(
                self.init_weights(nn.Linear(num_in, 128)),
                nn.LeakyReLU(),
                self.init_weights(nn.Linear(128, 128)),
                nn.LeakyReLU(),
            )
        else: # CNN input
            self.body = nn.Sequential(
                self.init_weights(nn.Conv2d(num_in[0], 32, 4, stride=2)),
                nn.LeakyReLU(),
                self.init_weights(nn.Conv2d(32, 64, 4, stride=2)),
                nn.LeakyReLU(),
                self.init_weights(nn.Conv2d(64, 64, 3, stride=2)),
                nn.LeakyReLU(),
                nn.Flatten(start_dim=1),
                self.init_weights(nn.Linear(39168, 512)),
                nn.LeakyReLU(),
            )          

        self.core = nn.Sequential(
            self.init_weights(nn.Linear(128+64+self.num_actions, 256)), 
            nn.LeakyReLU(),
            self.init_weights(nn.Linear(256, 256)),
            nn.LeakyReLU(),
        )
        self.rnn = nn.GRUCell(256, 64)
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.policy = nn.Sequential(
            self.init_weights(nn.Linear(256, 64), std=0.01),
            nn.LeakyReLU(),
            self.init_weights(nn.Linear(64, num_actions), std=0.01),
            nn.Softmax(dim=1)
        )

        self.value = nn.Sequential(
            self.init_weights(nn.Linear(256, 64), std=1),
            nn.LeakyReLU(),
            self.init_weights(nn.Linear(64, 1), std=1),
        )

        self.rnd_target = nn.Sequential(
            self.init_weights(nn.Linear(256, 64), std=1),
            nn.LeakyReLU(),
            self.init_weights(nn.Linear(64, 1), std=1),
        )

        self.rnd_pred = nn.Sequential(
            self.init_weights(nn.Linear(256, 64), std=1),
            nn.LeakyReLU(),
            self.init_weights(nn.Linear(64, 1), std=1),
        )

    def sample_noise(self):
        for layer in self.policy:
            if hasattr(layer, "sample_noise"):
                layer.sample_noise()
        for layer in self.value:
            if hasattr(layer, "sample_noise"):
                layer.sample_noise()

    def remove_noise(self):
        for layer in self.policy:
            if hasattr(layer, "remove_noise"):
                layer.remove_noise()
        for layer in self.value:
            if hasattr(layer, "remove_noise"):
                layer.remove_noise()

    def model_core(self, obs, last_action, h):
        if h is None:
            h = torch.zeros((obs.size(0), 64), device=obs.device)
        if last_action is None:
            last_action = torch.zeros((obs.size(0), self.num_actions), device=obs.device)

        body_out = self.body(obs)
        core_out = self.core(torch.concat((body_out, h, last_action), dim=1))
        h_new = self.rnn(core_out, h)
        return core_out, h_new
    
    def forward(self, obs, last_action, h):
        core_out, h_new = self.model_core(obs, last_action, h)
        return self.policy(core_out), self.value(core_out), self.rnd_target(core_out).detach(), self.rnd_pred(core_out.detach()), h_new
    
    #@torch.autocast(device_type="cuda")
    def get_action(self, obs, last_action, h):
        core_out, h_new = self.model_core(obs, last_action, h)
        policy_out = self.policy(core_out)
        return policy_out, h_new
    
    def get_value(self, obs, last_action, h):
        core_out, h_new = self.model_core(obs, last_action, h)
        value_out = self.value(core_out)
        return value_out, h_new

    def get_intrinsec_reward(self, obs, last_action, h):
        core_out, h_new = self.model_core(obs, last_action, h)
        rnd_target = self.rnd_target(core_out.detach()).detach()
        rnd_pred = self.rnd_pred(core_out)
        return torch.mean((rnd_target - rnd_pred).pow(2), dim=1), h_new
    
    def init_weights(self, layer, std=np.sqrt(2), bias=0.0):
        nn.init.orthogonal_(layer.weight, std)
        layer.bias.data.fill_(bias)
        return layer

class TensorMemory:
    def __init__(self, num_samples, state_size, num_actions):
        self.state = torch.zeros((num_samples, *(state_size)), dtype=torch.float32)
        self.hidden_state = torch.zeros((num_samples, 64), dtype=torch.float32)
        self.action = torch.zeros(num_samples, dtype=torch.long)
        self.last_action = torch.zeros((num_samples, num_actions), dtype=torch.float32)
        self.reward = torch.zeros(num_samples, dtype=torch.float32)
        self.next_state = torch.zeros((num_samples, *(state_size)), dtype=torch.float32)
        self.terminated = torch.zeros(num_samples, dtype=torch.bool)
        self.truncated = torch.zeros(num_samples, dtype=torch.bool)
        self.probs = torch.zeros((num_samples, num_actions), dtype=torch.float32)
        self.message

    def append(self, index, state, hidden_state, action, last_action, reward, next_state, terminated, truncated, probs):
        self.state[index] = state
        if hidden_state is not None:
            self.hidden_state[index] = hidden_state
        self.action[index] = action
        if last_action is not None:
            self.last_action[index] = last_action
        self.reward[index] = reward
        self.next_state[index] = next_state
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self.probs[index] = probs

class ListMemory:
    def __init__(self):
        self.state = []
        self.hidden_state = []
        self.action = []
        self.last_action = []
        self.reward = []
        self.next_state = []
        self.terminated = []
        self.truncated = []
        self.probs = []

    def append(self, index, state, hidden_state, action, last_action, reward, next_state, terminated, truncated, probs):
        self.state.append(state)
        if hidden_state is not None:
            self.hidden_state.append(hidden_state)
        else:
            self.hidden_state.append(torch.zeros((state.size(0), 64), dtype=torch.float32))
        self.action.append(action)
        if last_action is not None:
            self.last_action.append(last_action)
        self.reward.append(reward)
        self.next_state.append(next_state)
        self.terminated.append(terminated)
        self.truncated.append(truncated)
        self.probs.append(probs)


class PPO(nn.Module):
    def __init__(self, num_in, num_actions, env_buffer_size, gamma=0.99, writer=None):
        super(PPO, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.orig_model = ACNetwork(num_in, num_actions)
        self.model = self.orig_model # torch.compile(self.orig_model, mode="reduce-overhead")
        #self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, eps=1e-5)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4, fused=True)
        self.gamma = gamma
        self.eps_clip = 0.2
        self.memory = {}
        self.env_buffer_size = env_buffer_size
        if isinstance(num_in, int):           
            num_in = [num_in]
        self.state_size = num_in
        self.obs_max = nn.Parameter(torch.ones((num_in), dtype=torch.float32), requires_grad=False)
        #self.obs_max.data = torch.tensor([1.5, 1.5, 1.0, 1.0, 3.1415927, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        self.num_actions = num_actions
        self.writer = writer

    @torch.no_grad()
    def select_action(self, state, last_action, h, eval=False):
        state = torch.from_numpy(state).clone().float()
        if (state.dim() - len(self.state_size)) == 0:
            state = state.unsqueeze(0)
        state = state / self.obs_max
        state = state.to(self.device)
        if h is not None:
            h = h.to(self.device)
        if last_action is not None:
            last_action = torch.from_numpy(last_action).clone().float()
            last_action = last_action.to(self.device)
        self.model = self.model.to(self.device)
        probs, h_new = self.model.get_action(state, last_action, h)

        probs = probs.cpu()
        action = (probs.cumsum(-1) >= rand(probs.shape[:-1])[..., None]).byte().argmax(-1)
        if eval:
            action = probs.argmax()
        return action.cpu().detach().numpy(), probs.detach(), h_new.cpu().detach()
    
    @torch.no_grad()
    def get_intrinsic_reward(self, state, last_action, h):
        state = torch.from_numpy(state).clone().float()
        if (state.dim() - len(self.state_size)) == 0:
            state = state.unsqueeze(0)
        state = state / self.obs_max
        state = state.to(self.device)
        if h is not None:
            h = h.to(self.device)
        if last_action is not None:
            last_action = torch.from_numpy(last_action).clone().float()
            last_action = last_action.to(self.device)
        self.model = self.model.to(self.device)
        intrinsec_reward, h_new = self.model.get_intrinsec_reward(state, last_action, h)
        return intrinsec_reward.cpu().detach().numpy(), h_new.cpu().detach()
    
    def record_obs(self, state, hidden_state, action, last_action, reward, next_state, terminated, truncated, probs, env_id, step):
        # Define and if needed recalculate obs_max for all states
        state = torch.from_numpy(state)
        next_state = torch.from_numpy(next_state)
        terminated = torch.from_numpy(np.array(terminated))
        truncated = torch.from_numpy(np.array(truncated))

        if last_action is not None:
            last_action = torch.from_numpy(last_action)

        state = state / self.obs_max
        next_state = next_state / self.obs_max

        if env_id not in self.memory:
            if self.env_buffer_size is not None:
                self.memory[env_id] = TensorMemory(self.env_buffer_size, self.state_size, self.num_actions)
            else:
                self.memory[env_id] = ListMemory()
        self.memory[env_id].append(step, state, hidden_state, action, last_action, reward, next_state, terminated, truncated, probs)

    def train_epochs_bptt(self, optim_epoch):
        self.model = self.model.to(self.device)

        states, hidden_states, old_probs, actions, next_states, rewards, terminated, truncated, dones, loss_mask = self.prepareData()

        states = states.to(self.device)
        hidden_states = hidden_states.to(self.device)
        old_probs = old_probs.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        loss_mask = loss_mask.to(self.device)
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
        world_losses = []
        dones = dones.to(self.device)
        
        continue_training = True
        
        old_model = None
        for epoch in range(5):
            if epoch == 0:
                values = []
                next_values = []
                with torch.no_grad():
                    for i in range(self.env_buffer_size): # seq length, :]
                        probs, value, h, _ = self.model(states[:, i, :], hidden_states[:, i, :])
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
            ep_next_states = next_states.flatten(0,1)
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
                    ep_advantages[states_idx[:, i]] = (ep_advantages[states_idx[:, i]] - ep_advantages[states_idx[:, i]].mean()) / (ep_advantages[states_idx[:, i]].std() + 1e-8)

                    probs, pred_value, h, next_obs_pred = self.model(ep_states[states_idx[:, i]], h)
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

                    #pred_value_clipped = ep_returns[states_idx[:, i]] + torch.clamp(pred_value - ep_returns[states_idx[:, i]], -1.0, 1.0)
                    #value_loss_clipped = 0.5 * (ep_returns[states_idx[:, i]]-pred_value_clipped).pow(2)
                    value_loss = 0.5*(ep_returns[states_idx[:, i]]-pred_value).pow(2).mean()
                    #value_loss = torch.max(value_loss_clipped, value_loss).mean()
                    entropy_loss = -torch.clamp(ent, max=1.).mean()

                    world_loss = 0.5 * (next_obs_pred - ep_next_states[states_idx[:, i]]).pow(2).mean()

                    # Early stopping
                    with torch.no_grad():
                        log_ratio = log_probs - b_old_log_probs
                        approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        kl_divs.append(approx_kl_div)
                        if self.target_kl is not None and abs(approx_kl_div) > 1.5 * 0.01:
                            if old_model is not None:
                                self.model.load_state_dict(old_model)
                            continue_training = False
                            print(f"Early stopping due to reaching max kl: {approx_kl_div:.2f}, aprox kl: {approx_kl_div:.2f}")
                            break

                    #self.optimizer.zero_grad(set_to_none=True)
                    loss = (policy_loss + 0.5 *value_loss + 0 * entropy_loss) * loss_mask[:, i]
                    total_loss += loss / batch_seq_length
                    #loss = loss / batch_seq_length
                    #loss.backward(retain_graph=True)
                    #torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 0.5)
                    #self.optimizer.step()

                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    ents.append(ent.mean().item())
                    world_losses.append(world_loss.item())

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
            self.writer.add_scalar("train/clip_fraction", np.mean(clip_fractions), optim_epoch)
            self.writer.add_scalar("train/world_loss", sum(world_losses) / len(world_losses), optim_epoch)

    def train_epochs_bptt_2(self, optim_epoch):
        self.model = self.model.to(self.device)

        states, hidden_states, old_probs, actions, last_actions, next_states, rewards, terminated, truncated, dones, loss_mask = self.prepareData()

        states = states.to(self.device)
        hidden_states = hidden_states.to(self.device)
        old_probs = old_probs.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        loss_mask = loss_mask.to(self.device)
        
        batch_seq_length = states.size(1)//4

        policy_losses = []
        value_losses = []
        ents = []
        kl_divs=[]
        norms = []
        rnd_losses = []
        clip_fractions = []
        dones = dones.to(self.device)
        
        continue_training = True
        norms = []
        old_model = None
        for epoch in range(11):
            h = hidden_states[:, 0, :]
            with torch.no_grad():
                if epoch == 0 or epoch == 10 or continue_training is False or True:
                    values = []
                    next_values = []
                    for i in range(states.size(1)): # seq length, :]
                        probs, value, _, _, h = self.model(states[:, i, :].to(self.device), last_actions[:, i, :].to(self.device), hidden_states[:, i, :].to(self.device))
                        next_action = torch.zeros((states[:, i, :].size(0), self.num_actions), dtype=torch.float32, device=self.device)
                        next_action[torch.arange(states[:, i, :].size(0)), actions[:, i]] = 1
                        next_value, h_n = self.model.get_value(next_states[:, i, :].to(self.device), None, h)
                        
                        values.append(value.squeeze())
                        next_values.append(next_value.squeeze())

                        if i != (states.size(1)-1):
                            lm = loss_mask[:, i].bool()
                            done_mask = (dones[:, i] is True).to(self.device)
                            not_done_mask = (dones[:, i] is False).to(self.device)
                            h[done_mask | ~lm] = hidden_states[done_mask | ~lm, i+1, :].to(self.device)
                            hidden_states[not_done_mask | lm, i+1, :] = h[not_done_mask | lm]
                    

                    values = torch.stack(values, dim=1)
                    next_values = torch.stack(next_values, dim=1)
                    terminated = terminated.to(self.device)
                    truncated = truncated.to(self.device)
                    #if epoch == 0:
                    #    returns = self.calculate_returns(rewards, values, next_values, terminated, truncated, self.gamma)
                    #returns = returns.to(self.device)

                    if epoch == 0 or True:
                        advantages = self.calculate_advantages(rewards, values, next_values, terminated, truncated, self.gamma, 0.95)
                        if epoch == 0:
                            returns = advantages.cpu() + values.cpu()
                        advantages = (advantages - advantages[loss_mask.cpu()==1].mean()) / (advantages[loss_mask.cpu()==1].std() + 1e-8)
                        advantages = torch.clamp(advantages, torch.quantile(advantages[loss_mask.cpu()==1], 0.05), torch.quantile(advantages[loss_mask.cpu()==1], 0.95))
                    
                    returns = returns.to(self.device)
                    advantages = advantages.to(self.device)

                    if continue_training is False or epoch == 10:
                        break

            h = hidden_states[:, 0, :].to(self.device)
            total_loss = 0
            seq_log_probs = torch.zeros((states.size(0), batch_seq_length), dtype=torch.float32, device=self.device)
            seq_old_log_probs = torch.zeros((states.size(0), batch_seq_length), dtype=torch.float32, device=self.device)
            seq_mask = torch.zeros((states.size(0), batch_seq_length), dtype=torch.float32, device=self.device)
            for i in range(states.size(1)): # seq length, :]
                lm = loss_mask[:, i].to(self.device)

                probs, pred_value, rnd_target_values, rnd_pred_values, h = self.model(states[:, i, :].to(self.device), last_actions[:, i, :].to(self.device), h)
                pred_value = pred_value.squeeze()

                log_probs = torch.log(torch.gather(probs, -1, actions[:, i][..., None].to(self.device))) 
                ent = -(probs * torch.log(probs)).sum(dim=-1)
                log_probs = log_probs.squeeze()
                seq_log_probs[:, i%batch_seq_length] = log_probs

                b_old_probs = old_probs[:, i, :]
                b_old_log_probs = torch.log(torch.gather(b_old_probs, -1, actions[:, i][..., None])).squeeze().to(self.device)
                seq_old_log_probs[:, i%batch_seq_length] = b_old_log_probs

                # Set values outside the loss mask
                b_old_log_probs = torch.where(lm == 1, b_old_log_probs, log_probs) # Old probs are 0 outside the loss mask, set to probs to avoid nan
                pred_value = torch.where(lm == 1, pred_value, 0.0)

                ratios = torch.exp(log_probs - b_old_log_probs)
                surr1 = ratios * advantages[:, i].to(self.device)
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages[:, i].to(self.device)
                clip_fraction = torch.mean((torch.abs(ratios - 1) > self.eps_clip).float()).item()
                clip_fractions.append(clip_fraction)
                
                policy_loss = -(torch.min(surr1, surr2))[lm==1]
                value_loss = (returns[:, i].to(self.device)-pred_value).pow(2)[lm==1]
                entropy_loss = -torch.clamp(ent, max=1.)[lm==1]
                rnd_loss = (rnd_target_values.detach() - rnd_pred_values).pow(2)[lm==1]

                loss = policy_loss + 1.0 * value_loss + 0 * entropy_loss + 0.1 * rnd_loss
                loss = loss.mean() / batch_seq_length
                #loss.backward(retain_graph=True)
                total_loss += loss
                policy_losses.append(policy_loss.mean().cpu().item())
                value_losses.append(value_loss.mean().cpu().item())
                rnd_losses.append(rnd_loss.mean().cpu().item())
                ents.append(ent.mean().cpu().item())

                done_mask = (dones[:, i] is True).to(self.device).int()
                not_done_mask = (dones[:, i] is False).to(self.device).int()
                if i != states.size(1)-1:
                    h = h * not_done_mask[:, None] * lm[:, None] + hidden_states[:, i+1].to(self.device) * done_mask[:, None] * (~lm[:, None].bool()).int()
                
                seq_mask[:, i%batch_seq_length] = loss_mask[:, i].to(self.device)
                
                if (i+1)%batch_seq_length == 0:
                    # Early stopping
                    with torch.no_grad():
                        log_ratio = seq_log_probs.flatten() - seq_old_log_probs.flatten()
                        log_ratio = log_ratio[(seq_mask == 1).flatten()]
                        approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        kl_divs.append(approx_kl_div)
                        if abs(approx_kl_div) > 1.5 * 0.05:
                            if old_model is not None:
                                self.model.load_state_dict(old_model)
                            continue_training = False
                            print(f"Early stopping due to reaching max kl: {approx_kl_div:.2f}, aprox kl: {approx_kl_div:.2f}")
                            break

                    #old_model = self.model.state_dict()
                    total_loss.backward()
                    total_loss = 0
                    total_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    norms.append(total_norm)
                    h = h.detach()

                    torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        for i, key in enumerate(sorted(self.memory.keys())):
            self.memory[key].hidden_state = hidden_states[i, :len(self.memory[i].hidden_state)].cpu()

        self.update_obs_max()
                
        print("Policy Loss: " + str(sum(policy_losses) / len(policy_losses)) + " Value Loss: " + str(sum(value_losses) / len(value_losses)) + " Ent Loss: " + str(sum(ents) / len(ents)))
        if len(norms) > 0:
            print(f"{np.min(norms)} {np.mean(norms)} {np.max(norms)}")
        
        explained_var = explained_variance(values[loss_mask == 1].flatten().cpu().numpy(), returns[loss_mask == 1].flatten().cpu().numpy())
        if self.writer is not None:
            self.writer.add_scalar("train/policy_loss", sum(policy_losses) / len(policy_losses), optim_epoch)
            self.writer.add_scalar("train/value_loss", sum(value_losses) / len(value_losses), optim_epoch)
            self.writer.add_scalar("train/entropy_loss", sum(ents) / len(ents), optim_epoch)
            self.writer.add_scalar("train/rnd_loss", sum(rnd_losses) / len(rnd_losses), optim_epoch)
            self.writer.add_scalar("train/norms_min", np.min(norms), optim_epoch)
            self.writer.add_scalar("train/norms_mean", np.mean(norms), optim_epoch)
            self.writer.add_scalar("train/norms_max", np.max(norms), optim_epoch)
            self.writer.add_scalar("train/KL_div", np.mean(kl_divs), optim_epoch)
            self.writer.add_scalar("train/explained_var", explained_var, optim_epoch)
            self.writer.add_scalar("train/clip_fraction", np.mean(clip_fractions), optim_epoch)

    def prepareData(self):
        if type(self.memory[list(self.memory.keys())[0]]) == TensorMemory:
            states = torch.stack([self.memory[key].state for key in sorted(self.memory.keys())])
            hidden_states = torch.stack([self.memory[key].hidden_state for key in sorted(self.memory.keys())])
            old_probs = torch.stack([self.memory[key].probs for key in sorted(self.memory.keys())])
            actions = torch.stack([self.memory[key].action for key in sorted(self.memory.keys())])
            last_actions = torch.stack([self.memory[key].last_action for key in sorted(self.memory.keys())])
            next_states = torch.stack([self.memory[key].next_state for key in sorted(self.memory.keys())])
            rewards = torch.stack([self.memory[key].reward for key in sorted(self.memory.keys())])
            terminated = torch.stack([self.memory[key].terminated for key in sorted(self.memory.keys())])
            truncated = torch.stack([self.memory[key].truncated for key in sorted(self.memory.keys())])
            dones = torch.logical_or(terminated, truncated)
            loss_mask = torch.where(states.sum(dim=list(range(len(states.shape)))[2:]) == 0, 0.0, 1.0)
            return states, hidden_states, old_probs, actions, last_actions, next_states, rewards, terminated, truncated, dones, loss_mask
        elif type(self.memory[list(self.memory.keys())[0]]) == ListMemory:
            max_history_length = max([len(self.memory[key].state) for key in self.memory.keys()])
            states = torch.zeros((len(self.memory.keys()), max_history_length, *(self.state_size)), dtype=torch.float32)
            hidden_states = torch.zeros((len(self.memory.keys()), max_history_length, 64), dtype=torch.float32)
            old_probs = torch.zeros((len(self.memory.keys()), max_history_length, self.num_actions), dtype=torch.float32)
            actions = torch.zeros((len(self.memory.keys()), max_history_length), dtype=torch.long)
            last_actions = torch.zeros((len(self.memory.keys()), max_history_length, self.num_actions), dtype=torch.float32)
            next_states = torch.zeros((len(self.memory.keys()), max_history_length, *(self.state_size)), dtype=torch.float32)
            rewards = torch.zeros((len(self.memory.keys()), max_history_length), dtype=torch.float32)
            terminated = torch.zeros((len(self.memory.keys()), max_history_length), dtype=torch.bool)
            truncated = torch.zeros((len(self.memory.keys()), max_history_length), dtype=torch.bool)
            loss_mask = torch.zeros((len(self.memory.keys()), max_history_length), dtype=torch.float32)
            for i, key in enumerate(sorted(self.memory.keys())):
                states[i, :len(self.memory[key].state)] = torch.stack(self.memory[key].state)
                hidden_states[i, :len(self.memory[key].state)] = torch.stack(self.memory[key].hidden_state).squeeze()
                old_probs[i, :len(self.memory[key].state)] = torch.stack(self.memory[key].probs).squeeze()
                actions[i, :len(self.memory[key].state)] = torch.from_numpy(np.asarray(self.memory[key].action)).squeeze()
                last_actions[i, :len(self.memory[key].state)] = torch.stack(self.memory[key].last_action).squeeze()
                next_states[i, :len(self.memory[key].state)] = torch.stack(self.memory[key].next_state).squeeze()
                rewards[i, :len(self.memory[key].state)] = torch.from_numpy(np.asarray(self.memory[key].reward)).squeeze()
                terminated[i, :len(self.memory[key].state)] = torch.from_numpy(np.asarray(self.memory[key].terminated)).squeeze()
                truncated[i, :len(self.memory[key].state)] = torch.from_numpy(np.asarray(self.memory[key].truncated)).squeeze()
                loss_mask[i, :len(self.memory[key].state)] = 1
            dones = torch.logical_or(terminated, truncated)
            return states, hidden_states, old_probs, actions, last_actions, next_states, rewards, terminated, truncated, dones, loss_mask

    
    def save_model(self, path):
        torch.save(self.state_dict(), path + ".pt")

    def load_model(self, path):
        self.load_state_dict(torch.load(path + ".pt"))

    def update_obs_max(self):
        states, hidden_states, old_probs, actions, last_actions, next_states, rewards, terminated, truncated, dones, loss_mask = self.prepareData()
        states = states * self.obs_max

        old_obs_max = self.obs_max.clone()
        states_obs_max = torch.max(torch.max(states, 0)[0], 0)[0]
        self.obs_max.data = torch.max(torch.stack((self.obs_max, states_obs_max)), axis=0)[0]
        self.obs_max.data = self.obs_max.data.type(torch.float32)

        if (self.obs_max != old_obs_max).max() == 1:
            for id in self.memory.keys():
                self.memory[id].state = self.memory[id].state * old_obs_max / self.obs_max
                self.memory[id].next_state = self.memory[id].next_state * old_obs_max / self.obs_max
    
    def calculate_returns(self, rewards, values, next_values, terminated, truncated, discount_factor):
        dones = torch.logical_or(terminated, truncated)
        R = next_values[:, -1] * (~terminated[:, -1]).int()
        returns = torch.zeros(rewards.shape)

        for t in reversed(range(rewards.size(1))):
            R = rewards[:, t] + R * discount_factor * (~dones[:, t]).int() + next_values[:, t] * truncated[:, t].int()
            returns[:, t] = R
        return returns
    
    def calculate_advantages(self, rewards, values, next_values, terminated, truncated, discount_factor, trace_decay):
        advantages = torch.zeros(rewards.shape)
        dones = torch.logical_or(terminated, truncated)
        adv = (next_values[:, -1] - values[: , -1]) * (~terminated[:, -1]).int()

        for t in reversed(range(rewards.size(1))):
            adv = adv * (~dones[:, t]).int()
            delta = rewards[:, t] + (discount_factor * next_values[: ,t] * (~terminated[:, t]).int()) - values[:, t]
            adv = delta + discount_factor * trace_decay * adv
            advantages[:, t] = adv

        return advantages
        
    
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y