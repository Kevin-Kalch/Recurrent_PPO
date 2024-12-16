from random import Random
import numpy as np
import torch
import torch.nn as nn
import math
from torch import rand

# https://github.com/Kaixhin/NoisyNet-A3C/blob/master/model.py
class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(
            in_features, out_features, bias=True
        )  # TODO: Adapt for no bias
        # µ^w and µ^b reuse self.weight and self.bias
        self.sigma_init = sigma_init
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features))  # σ^w
        self.sigma_bias = nn.Parameter(torch.Tensor(out_features))  # σ^b
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(
            self, "sigma_weight"
        ):  # Only init after all params added (otherwise super().__init__() fails)
            nn.init.uniform_(
                self.weight,
                -math.sqrt(3 / self.in_features),
                math.sqrt(3 / self.in_features),
            )
            nn.init.uniform_(
                self.bias,
                -math.sqrt(3 / self.in_features),
                math.sqrt(3 / self.in_features),
            )
            nn.init.constant_(self.sigma_weight, self.sigma_init)
            nn.init.constant_(self.sigma_bias, self.sigma_init)

    def forward(self, input):
        return torch.nn.functional.linear(
            input,
            self.weight + self.sigma_weight * self.epsilon_weight,
            self.bias + self.sigma_bias * self.epsilon_bias,
        )

    def sample_noise(self):
        self.epsilon_weight = torch.randn(self.out_features, self.in_features)
        self.epsilon_bias = torch.randn(self.out_features)

    def remove_noise(self):
        self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
        self.epsilon_bias = torch.zeros(self.out_features)

# https://arxiv.org/pdf/1706.01905
class ACNetwork(nn.Module):
    def __init__(self, num_in, num_actions, config) -> None:
        super(ACNetwork, self).__init__()
        self.num_in = num_in
        self.num_actions = num_actions
        self.config = config

        if isinstance(num_in, int): # Linear input
            self.body = nn.Sequential(
                self.init_weights(nn.Linear(num_in, 256)),
                nn.LeakyReLU(),
                self.init_weights(nn.Linear(256, 256)),
                nn.LeakyReLU(),
                self.init_weights(nn.Linear(256, 128)),
                nn.LeakyReLU(),
            )
        else:
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
            self.init_weights(nn.Linear(128+self.config["hidden_size"]+self.num_actions, 256)), 
            nn.LeakyReLU(),
            self.init_weights(nn.Linear(256, 256)),
            nn.LeakyReLU(),
        )
        self.rnn = nn.GRUCell(256, self.config["hidden_size"])
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.policy = nn.Sequential(
            self.init_weights(nn.Linear(256, 64)),
            nn.LeakyReLU(),
            self.init_weights(NoisyLinear(64, num_actions), std=0.1),
            nn.Softmax(dim=1)
        )

        self.value = nn.Sequential(
            self.init_weights(nn.Linear(256, 64)),
            nn.LeakyReLU(),
            self.init_weights(nn.Linear(64, 1), std=1),
        )

        self.rnd_target = nn.Sequential(
            self.init_weights(nn.Linear(256, 64)),
            nn.LeakyReLU(),
            self.init_weights(nn.Linear(64, 1), std=1),
        )

        self.rnd_pred = nn.Sequential(
            self.init_weights(nn.Linear(256, 64)),
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
        if h is None or self.config["use_memory"] == False:
            h = torch.zeros((obs.size(0), self.config["hidden_size"]), device=obs.device)
        if last_action is None:
            last_action = torch.zeros((obs.size(0), self.num_actions), device=obs.device)

        body_out = self.body(obs)
        core_out = self.core(torch.concat((body_out, h, last_action), dim=1))
        return core_out
    
    def forward(self, obs, last_action, h):
        core_out = self.model_core(obs, last_action, h)
        return self.policy(core_out), self.value(core_out), self.rnd_target(core_out).detach(), self.rnd_pred(core_out.detach()), self.rnn(core_out, h)
    
    #@torch.autocast(device_type="cuda")
    def get_action(self, obs, last_action, h):
        core_out = self.model_core(obs, last_action, h)
        policy_out = self.policy(core_out)
        return policy_out, self.rnn(core_out, h)
    
    def get_value(self, obs, last_action, h):
        core_out = self.model_core(obs, last_action, h)
        value_out = self.value(core_out)
        return value_out, self.rnn(core_out, h)

    def get_intrinsec_reward(self, obs, last_action, h):
        core_out = self.model_core(obs, last_action, h)
        rnd_target = self.rnd_target(core_out.detach()).detach()
        rnd_pred = self.rnd_pred(core_out)
        return torch.mean((rnd_target - rnd_pred).pow(2), dim=1), self.rnn(core_out, h)
    
    def init_weights(self, layer, std=np.sqrt(2), bias=0.0):
        nn.init.orthogonal_(layer.weight, std)
        #nn.init.kaiming_normal_(layer.weight, 0.01, nonlinearity='leaky_relu')
        layer.bias.data.fill_(bias)
        return layer

class TensorMemory:
    def __init__(
        self, num_samples, state_size, num_actions, config
    ) -> None:
        self.state = torch.zeros(
            (num_samples, *(state_size)), dtype=torch.float32
        )
        self.hidden_state = torch.zeros(
            (num_samples, config["hidden_size"]), dtype=torch.float32
        )
        self.action = torch.zeros(num_samples, dtype=torch.long)
        self.last_action = torch.zeros(
            (num_samples, num_actions), dtype=torch.float32
        )
        self.reward = torch.zeros(num_samples, dtype=torch.float32)
        self.next_state = torch.zeros(
            (num_samples, *(state_size)), dtype=torch.float32
        )
        self.terminated = torch.zeros(num_samples, dtype=torch.bool)
        self.truncated = torch.zeros(num_samples, dtype=torch.bool)
        self.probs = torch.zeros(
            (num_samples, num_actions), dtype=torch.float32
        )

        self.action_required = torch.zeros(num_samples, dtype=torch.bool)
        self.weight = torch.ones(num_samples, dtype=torch.float32)

    def append(
        self,
        timestep,
        state,
        hidden_state,
        action,
        last_action,
        reward,
        next_state,
        terminated,
        truncated,
        probs,
        action_required,
    ):
        self.state[timestep] = state
        if hidden_state is not None:
            self.hidden_state[timestep] = hidden_state
        self.action[timestep] = action
        if last_action is not None:
            self.last_action[timestep] = last_action
        self.reward[timestep] = reward
        self.next_state[timestep] = next_state
        self.terminated[timestep] = terminated
        self.truncated[timestep] = truncated
        self.probs[timestep] = probs
        self.action_required[timestep] = action_required

class ListMemory:
    def __init__(self, config):
        self.state = []
        self.hidden_state = []
        self.action = []
        self.last_action = []
        self.reward = []
        self.next_state = []
        self.terminated = []
        self.truncated = []
        self.probs = []
        self.action_required = []
        self.weight = []
        self.config = config

    def append(
            self,
            timestep,
            state,
            hidden_state,
            action,
            last_action,
            reward,
            next_state,
            terminated,
            truncated,
            probs,
            action_required,
        ):
        self.state.append(state)
        if hidden_state is not None:
            self.hidden_state.append(hidden_state)
        else:
            self.hidden_state.append(torch.zeros((state.size(0), self.config["hidden_size"]), dtype=torch.float32))
        self.action.append(action)
        if last_action is not None:
            self.last_action.append(last_action)
        self.reward.append(reward)
        self.next_state.append(next_state)
        self.terminated.append(terminated)
        self.truncated.append(truncated)
        self.probs.append(probs)
        self.action_required.append(action_required)
        self.weight.append(1.0)


class PPO(nn.Module):
    def __init__(self, num_in, num_actions, config, writer):
        super(PPO, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.model = ACNetwork(num_in, num_actions, config)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config["lr"], fused=True
        )
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     self.optimizer,
        #     base_lr=config["base_lr"],
        #     max_lr=config["max_lr"],
        #     step_size_up=config["step_size_up"],
        #     cycle_momentum=False,
        # )
        self.gamma = config["gamma"]
        self.eps_clip = config["eps_clip"]
        self.memory = {}
        self.env_buffer_size = config["steps_per_env"]
        if isinstance(num_in, int):           
            num_in = [num_in]
        self.state_size = num_in
        self.obs_max = nn.Parameter(
            torch.ones((num_in), dtype=torch.float32), requires_grad=False
        )
        self.num_actions = num_actions
        self.config = config
        self.writer = writer

    @torch.no_grad()
    def select_action(self, state, last_action, h, eval=False):
        state = torch.from_numpy(state).clone().float()
        if (state.dim() - len(self.state_size)) == 0:
            state = state.unsqueeze(0)
            h = h.unsqueeze(0)
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
            action = probs.argmax(-1)
        return action.cpu().detach().numpy(), probs.detach(), h_new.cpu().detach()
    
    @torch.no_grad()
    def get_intrinsic_reward(self, state, last_action, h):
        state = torch.from_numpy(state).clone().float()
        if (state.dim() - len(self.state_size)) == 0:
            state = state.unsqueeze(0)
            h = h.unsqueeze(0)
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
    
    def record_obs(self, state, hidden_state, action, last_action, reward, next_state, terminated, truncated, probs, action_required, env_id, step):
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
                self.memory[env_id] = TensorMemory(self.env_buffer_size, self.state_size, self.num_actions, self.config)
            else:
                self.memory[env_id] = ListMemory(self.config)
        self.memory[env_id].append(step, state, hidden_state, action, last_action, reward, next_state, terminated, truncated, probs, action_required)

    def train_epochs_bptt(self, optim_epoch):
        self.model = self.model.to(self.device)

        states, hidden_states, old_probs, actions, last_actions, next_states, rewards, terminated, truncated, dones, loss_mask, actions_required, weights = self.prepareData()
        current_game_policy_probs = torch.zeros_like(old_probs)
        batch_seq_length = states.size(1) // self.config["batches_per_sequence"]

        policy_losses = []
        value_losses = []
        ents = []
        kl_divs = []
        norms = []
        clip_fractions = []
        rnd_losses = []
        
        continue_training = True
        
        old_model = None
        for epoch in range(self.config["ppo_epochs"] + 1):
            h = hidden_states[:, 0, :].to(self.device)
            active_policy_probs = torch.zeros_like(old_probs)
            with torch.no_grad():
                if (
                    epoch == 0
                    or epoch == self.config["ppo_epochs"]
                    or continue_training == False
                    or self.config["recalculate_returns"]
                    or self.config["recalculate_advantages"]
                ):
                    values = []
                    next_values = []
                
                    for i in range(states.size(1)): # seq length, :]
                        state = states[:, i, :].to(self.device)
                        last_action = last_actions[:, i, :].to(self.device)

                        probs, value, _, _, h = self.model(state, last_action, h)
                        if epoch == 0:
                            current_game_policy_probs[:, i, :] = probs.cpu()
                        active_policy_probs[:, i, :] = probs.cpu()

                        next_state = next_states[:, i, :].to(self.device)

                        if i != (states.size(1)-1):
                            next_last_action = last_actions[:, i+1, :].to(self.device)
                        else:
                            next_last_action = torch.zeros_like(last_action)
                            next_last_action[:, actions[:, i].to(self.device)] = 1

                        if last_actions.sum() == 0:
                            next_last_action = None
                        next_value, h_n = self.model.get_value(next_state, next_last_action, h)
                        
                        values.append(value.squeeze())
                        next_values.append(next_value.squeeze())

                        if i != (states.size(1)-1):
                            lm = loss_mask[:, i].bool().to(self.device)
                            done_mask = (dones[:, i] == True).to(self.device)
                            not_done_mask = (dones[:, i] == False).to(self.device)

                            h[torch.logical_or(done_mask, ~lm)] = hidden_states[torch.logical_or(done_mask.cpu(), ~lm.cpu())][:, i + 1, :].to(self.device)
                            hidden_states[torch.logical_and(not_done_mask.cpu(), lm.cpu())][:, i + 1, :] = h[torch.logical_and(not_done_mask, lm)].cpu()

                    values = torch.stack(values, dim=1).cpu()
                    next_values = torch.stack(next_values, dim=1).cpu()

                    if epoch == 0 or self.config["recalculate_returns"]:
                            returns = self.calculate_returns(
                                rewards,
                                values,
                                next_values,
                                terminated,
                                truncated,
                                self.gamma,
                            )

                    if epoch == 0 or self.config["recalculate_advantages"] or continue_training == False or epoch == self.config["ppo_epochs"]:
                        advantages = self.calculate_advantages(
                            rewards,
                            values,
                            next_values,
                            terminated,
                            truncated,
                            self.gamma,
                            0.95,
                        )
                        adv, rets = self.calculate_vtrace_advantages(
                            rewards,
                            values,
                            next_values,
                            old_probs,
                            active_policy_probs,
                            actions,
                            terminated,
                            truncated,
                            self.gamma,
                            0.95,
                            1.0,
                        )
                        # if epoch == 0 or self.config["recalculate_returns"]:
                        #     returns = rets
                        #     returns = adv + values
                        # if epoch == 0 or self.config["recalculate_advantages"]:
                        #     advantages = adv
                        rets = self.calculate_v_trace_returns_impala(
                            rewards,
                            values,
                            next_values,
                            old_probs,
                            active_policy_probs,
                            actions,
                            terminated,
                            truncated,
                            self.gamma,
                            0.95,
                            1.0,
                        )

                        geppo_advs = []
                        geppo_returns = []
                        for run_id in range(values.size(0)):
                            gadv, gret = self.gae_vtrace(
                                actions[run_id],
                                next_values[run_id],
                                rewards[run_id],
                                terminated[run_id],
                                truncated[run_id],
                                active_policy_probs[run_id],
                                old_probs[run_id],
                                self.gamma,
                                0.95,
                                1.0,
                                values[run_id],
                            )
                            geppo_advs.append(gadv)
                            geppo_returns.append(gret)
                        geppo_advs = torch.stack(geppo_advs, dim=0)
                        geppo_returns = torch.stack(geppo_returns, dim=0)
                        if epoch == 0 or self.config["recalculate_returns"]:
                            returns = geppo_returns
                        if epoch == 0 or self.config["recalculate_advantages"]:
                            advantages = geppo_advs
                        # if epoch == 0 or self.config["recalculate_returns"]:
                        #      returns = rets
                        # advantages = (advantages - advantages[loss_mask == 1].mean()) / (advantages[loss_mask == 1].std() + 1e-8)
                        # advantages = torch.clamp(advantages, torch.quantile(advantages[loss_mask.cpu()==1], 0.05), torch.quantile(advantages[loss_mask.cpu()==1], 0.95))

                         # Minibatch adv
                        current_policy_log_prob = torch.log(
                            torch.gather(current_game_policy_probs, -1, actions[..., None])
                        ).squeeze()
                        old_log_probs = torch.log(
                            torch.gather(old_probs, -1, actions[..., None])
                        ).squeeze()
                        active_policy_log_prob = torch.log(
                            torch.gather(active_policy_probs, -1, actions[..., None])
                        ).squeeze()

                        offpol_ratio = torch.exp(current_policy_log_prob - old_log_probs)
                        ratios = torch.exp(active_policy_log_prob - old_log_probs)
                        adv_mean = (advantages * offpol_ratio * weights).mean() / (offpol_ratio * weights).mean()
                        adv_std = (advantages * offpol_ratio * weights).std() + 1e-8
                        adv_mean = adv_mean.detach()
                        adv_std = adv_std.detach()

                        #adv_mean = advantages[loss_mask == 1].mean()
                        #adv_std = advantages[loss_mask == 1].std() + 1e-8

                        ratio_diff = torch.abs(ratios - offpol_ratio)
                        tv = 0.5 * (weights * ratio_diff).pow(2).mean()
                        if tv > (0.5*self.eps_clip):
                            print("GePPO early stop")
                            continue_training = False
                            break
                    if continue_training == False or epoch == self.config["ppo_epochs"]:
                        break

            h = hidden_states[:, 0, :].to(self.device)
            h.requires_grad = True
            total_loss = 0
            seq_log_probs = torch.zeros(
                (states.size(0), batch_seq_length), dtype=torch.float32, device=self.device
            )
            seq_old_log_probs = torch.zeros(
                (states.size(0), batch_seq_length), dtype=torch.float32, device=self.device
            )
            seq_mask = torch.zeros(
                (states.size(0), batch_seq_length), dtype=torch.float32, device=self.device
            )
            for i in range(states.size(1)): # seq length, :]
                lm = loss_mask[:, i].to(self.device)
                actn_req = actions_required[:, i].to(self.device)

                if lm.sum() > 0:
                    state = states[:, i, :].to(self.device)
                    last_action = last_actions[:, i, :].to(self.device)
                    current_policy_prob = current_game_policy_probs[:, i, :]
                    b_weights = weights[:, i].to(self.device)
                    probs, pred_value, rnd_target_values, rnd_pred_values, h = self.model(state, last_action, h)

                    pred_value = pred_value.squeeze()
                    pred_value = torch.where(lm == 1, pred_value, 0.0)

                    log_probs = torch.log(
                        torch.gather(probs, -1, actions[:, i][..., None].to(self.device))
                    )
                    ent = -(probs * torch.log(probs)).sum(dim=-1)
                    log_probs = log_probs.squeeze()
                    seq_log_probs[:, i % batch_seq_length] = log_probs

                    b_old_probs = old_probs[:, i, :]
                    b_old_log_probs = torch.log(
                        torch.gather(b_old_probs, -1, actions[:, i][..., None])
                    ).squeeze().to(self.device)
                    seq_old_log_probs[:, i % batch_seq_length] = b_old_log_probs

                    # Set values outside the loss mask
                    b_old_log_probs = torch.where(
                        lm == 1, b_old_log_probs, log_probs
                    )  # Old probs are 0 outside the loss mask, set to probs to avoid nan

                    ent = -(probs * torch.log(probs)).sum(dim=-1)
                    ent_mask = torch.isfinite(ent)
                    adv = advantages[:, i].to(self.device)

                    valid_point_mask = torch.logical_and(torch.logical_and(lm, ent_mask), actn_req)

                    ratios = torch.exp(log_probs - b_old_log_probs)
                    current_policy_log_prob = torch.log(
                        torch.gather(current_policy_prob, -1, actions[:, i][..., None])
                    ).squeeze().to(self.device)
                    seq_old_log_probs[:, i % batch_seq_length] = current_policy_log_prob
                    clip_mean = torch.exp(current_policy_log_prob - b_old_log_probs)
                    KL = (probs * (torch.log(probs) - torch.log(b_old_probs.to(self.device)))).sum(dim=-1)

                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    surr1 = ratios * adv
                    surr2 = torch.clamp(ratios, clip_mean - self.eps_clip, clip_mean + self.eps_clip) * adv
                    clip_fraction = (torch.abs(ratios[valid_point_mask] - 1) > self.eps_clip).float().mean()
                    clip_fractions.append(clip_fraction.item())

                    if actn_req.sum() > 0:
                        if not self.config["use_truly_ppo"]:
                            policy_loss = -torch.min(surr1, surr2)[valid_point_mask]
                        else:
                            policy_loss = -torch.where(
                                (KL >= self.config["max_kl_div"]) & (ratios * adv > clip_mean * adv), # 1 for ratios of old policy to old policy
                                ratios * adv - self.config["policy_slope"] * KL,
                                ratios * adv - self.config["max_kl_div"]
                            )[valid_point_mask]
                        
                        value_loss = (returns[:, i].to(self.device)-pred_value).pow(2)[valid_point_mask]
                        ent_loss = -torch.clamp(ent, max=1.0)[valid_point_mask] 
                        rnd_loss = (rnd_target_values - rnd_pred_values).pow(2)[valid_point_mask]
                        loss = (
                            self.config["policy_weight"] * policy_loss
                            + self.config["value_weight"] * value_loss
                            + self.config["entropy_weight"] * ent_loss
                            + self.config["rnd_weight"] * rnd_loss
                        ) * b_weights[valid_point_mask]
                        loss = loss.mean() / batch_seq_length
                        total_loss += loss

                        policy_losses.append(policy_loss.mean().cpu().item())
                        value_losses.append(value_loss.mean().cpu().item())
                        ents.append(ent_loss.mean().cpu().item())
                        rnd_losses.append(rnd_loss.mean().cpu().item())
                
                done_mask = (dones[:, i] == True).to(self.device).int().reshape(-1)
                not_done_mask = (dones[:, i] == False).to(self.device).int().reshape(-1)
                if i != (states.size(1)-1):
                    h = h * not_done_mask[:, None] * lm[:, None] + hidden_states[not_done_mask.cpu()][:, i + 1, :].to(self.device) * not_done_mask[:, None] * (~lm[:, None].bool()).int()
                
                seq_mask[:, i % batch_seq_length] = torch.logical_and(lm, actn_req).float()

                if (i + 1) % batch_seq_length == 0:
                    with torch.no_grad():
                        log_ratio = seq_log_probs - seq_old_log_probs
                        log_ratio = log_ratio[seq_mask == 1]
                        approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy().item()
                        kl_divs.append(approx_kl_div)
                        if (
                            abs(approx_kl_div) > 1.5 * self.config["max_kl_div"]
                            and self.config["early_stopping"]
                        ):
                            if old_model is not None and self.config["es_restore_model"]:
                                self.model.load_state_dict(old_model)
                            continue_training = False
                            print(
                                f"Early stopping due to reaching max kl: {approx_kl_div:.2f}, aprox kl: {approx_kl_div:.2f}"
                            )
                            break
                    old_model = self.model.state_dict()
                    self.optimizer.zero_grad()
                    total_loss.backward(retain_graph=True)
                    total_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1.0 / 2)
                    norms.append(total_norm)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.config["use_scheduler"]:
                        self.scheduler.step()
                    total_loss = 0
                    h = h.detach()
        
        for i, key in enumerate(sorted(self.memory.keys())):
            self.memory[key].hidden_state = hidden_states[i].cpu()

        if self.config["use_obs_max"]:
            self.update_obs_max()

        self.adapt_factor = 0.03
        self.adapt_maxthresh = 1.0
        self.adapt_minthresh = 0.4
        # if tv > (self.adapt_maxthresh * (0.5*self.eps_clip)):
        #     for param_group in self.optimizer.param_groups:
        #         lr_new = (param_group["lr"] / 
        #             (1+self.adapt_factor))
        #         param_group['lr'] = lr_new

        #     if self.writer is not None:
        #         self.writer.add_scalar("train/lr", lr_new, optim_epoch)
        # elif tv < (self.adapt_minthresh * (0.5*self.eps_clip)):
        #     for param_group in self.optimizer.param_groups:
        #         lr_new = (param_group["lr"] * 
        #             (1+self.adapt_factor))
        #         param_group['lr'] = lr_new
            
        #     if self.writer is not None:
        #         self.writer.add_scalar("train/lr", lr_new, optim_epoch)
                
        print(
            "Policy Loss: "
            + str(sum(policy_losses) / len(policy_losses))
            + " Value Loss: "
            + str(sum(value_losses) / len(value_losses))
            + " Ent Loss: "
            + str(sum(ents) / len(ents))
        )

        if len(norms) > 0:
            print(f"{np.min(norms)} {np.mean(norms)} {np.max(norms)}")

        explained_var = explained_variance(values[loss_mask == 1].flatten().cpu().numpy(), returns[loss_mask == 1].flatten().cpu().numpy())
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
            self.writer.add_scalar("train/tv", tv, optim_epoch)

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
            actions_required = torch.stack([self.memory[key].action_required for key in sorted(self.memory.keys())])
            weights = torch.stack([self.memory[key].weight for key in sorted(self.memory.keys())])
            dones = torch.logical_or(terminated, truncated)
            loss_mask = torch.where(torch.tensor(np.isclose(states.sum(dim=list(range(len(states.shape)))[2:]), 0)), 0.0, 1.0)
            return states, hidden_states, old_probs, actions, last_actions, next_states, rewards, terminated, truncated, dones, loss_mask, actions_required, weights
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
            actions_required = torch.zeros((len(self.memory.keys()), max_history_length), dtype=torch.bool)
            weights = torch.ones((len(self.memory.keys()), max_history_length), dtype=torch.float32)
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
                loss_mask[i, :len(self.memory[key].state)] = torch.from_numpy(np.isclose((np.asarray(self.memory[key].state)).sum(dim=list(range(len(states.shape)))[2:]), 0)).squeeze()
                actions_required[i, :len(self.memory[key].state)] = torch.from_numpy(np.asarray(self.memory[key].action_required)).squeeze()
                weights[i, :len(self.memory[key].state)] = torch.from_numpy(np.asarray(self.memory[key].weight)).squeeze()
            dones = torch.logical_or(terminated, truncated)
            return states, hidden_states, old_probs, actions, last_actions, next_states, rewards, terminated, truncated, dones, loss_mask, actions_required, weights

    
    def save_model(self, path):
        torch.save(self.state_dict(), path + ".pt")

    def load_model(self, path):
        self.load_state_dict(torch.load(path + ".pt"))

    def update_obs_max(self):
        states, hidden_states, old_probs, actions, last_actions, next_states, rewards, terminated, truncated, dones, loss_mask, actions_required, weights = self.prepareData()
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
    
    def calculate_vtrace_advantages(self, rewards, values, next_values, old_probs, current_probs, actions, terminated, truncated, discount_factor, trace_decay, clip_rho_threshold):
        advantages = torch.zeros(rewards.shape)
        returns = torch.zeros(rewards.shape)
        dones = torch.logical_or(terminated, truncated)

        rho = torch.ones_like(rewards[:, -1])
        adv = (next_values[:, -1] - values[: , -1]) * (~terminated[:, -1]).int()
        for t in reversed(range(rewards.size(1))):
            adv = adv * (~dones[:, t]).int()

            old_action_prob = torch.gather(old_probs[:, -1], -1, actions[:, t][..., None]).squeeze()
            action_prob = torch.gather(current_probs[:, -1], -1, actions[:, t][..., None]).squeeze()
            current_rho = torch.min(torch.fill(torch.ones_like(action_prob), clip_rho_threshold), action_prob / old_action_prob)
            rho = torch.where(dones[:, t], current_rho, rho * current_rho)
            
            delta = (rewards[:, t] + (discount_factor * next_values[:, t] * (~terminated[:, -1]).int()) - values[:, t])
            advantages[:, t] = (delta + discount_factor * adv * rho)
            returns[:, t] = rewards[:, t] + discount_factor * adv * rho
            adv = advantages[:, t]
        return advantages, returns
    
    def calculate_v_trace_returns_impala(self, rewards, values, next_values, old_probs, current_probs, actions, terminated, truncated, discount_factor, trace_decay, clip_rho_threshold):
        dones = torch.logical_or(terminated, truncated)
        R = next_values[:, -1] * (~terminated[:, -1]).int()
        returns = torch.zeros(rewards.shape)

        for t in reversed(range(rewards.size(1))):
            R = R * (~dones[:, t]).int() + next_values[:, t] * truncated[:, t].int()

            old_action_prob = torch.gather(old_probs[:, -1], -1, actions[:, t][..., None]).squeeze()
            action_prob = torch.gather(current_probs[:, -1], -1, actions[:, t][..., None]).squeeze()
            c = trace_decay * torch.clamp(action_prob/old_action_prob, max=1.0)
            deltaV = torch.clamp(action_prob/old_action_prob, max=1.0) * (rewards[:, t] + discount_factor * next_values[:, t] * (~terminated[:, t]).int() - values[:, t])
            R = values[:, t] + deltaV + discount_factor * c * (R - next_values[:, t]) * (~terminated[:, t]).int()
            returns[:, t] = R
        return returns
        
    def gae_vtrace(self, actions,next_values,rewards,terminated, truncated, current_probs, old_probs,gamma,lam,is_trunc, values):
        """Calculates off-policy GAE with V-trace for trajectory."""
        dones = torch.logical_or(terminated, truncated)
        neglogp_pik = torch.gather(old_probs, -1, actions[..., None]).squeeze()
        action_prob = torch.gather(current_probs, -1, actions[..., None]).squeeze()

        ratio = np.exp(action_prob - neglogp_pik)
        ratio_trunc = np.minimum(ratio,is_trunc)

        n = ratio.shape[0]
        ones_U = np.triu(np.ones((n,n)),0)
        
        rate_L = np.tril(np.ones((n,n))*gamma*lam,-1)
        rates = np.tril(np.cumprod(rate_L+ones_U,axis=0),0)

        ratio_trunc_repeat = np.repeat(np.expand_dims(ratio_trunc,1),n,axis=1)
        ratio_trunc_L = np.tril(ratio_trunc_repeat,-1)
        ratio_trunc_prods = np.tril(np.cumprod(ratio_trunc_L+ones_U,axis=0),0)

        V = values
        Vp = next_values

        delta = rewards + gamma * (1-terminated.int()) * Vp - V

        intermediate = rates * ratio_trunc_prods * np.expand_dims(delta,axis=1)
        adv = torch.sum(torch.from_numpy(intermediate) ,axis=0)
        rtg = adv * ratio_trunc + V

        adv = adv.float()
        rtg = rtg.float()

        return adv, rtg
    
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y