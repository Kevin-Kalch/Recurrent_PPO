from collections import deque
import gymnasium
import torch
from model import PPO
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from lunar_lander_helpers import VelHidden, LastAction, TruncationPenalty
from gymnasium.wrappers import TimeAwareObservation

torch.manual_seed(4020)
#import flappy_bird_gymnasium

def create_envs(num=4, test=False):
    envs = []
    for i in range(num):
        def gen():
            env = LastAction(VelHidden(gymnasium.make('LunarLander-v2')))
            env = TimeAwareObservation(LastAction(env))
            env = gymnasium.wrappers.RecordEpisodeStatistics(env)
            return env
        envs.append(gen)
    if num == 1:
        return envs[0]()
    envs = gymnasium.vector.SyncVectorEnv(envs)
    return envs

def train(config, writer: SummaryWriter = None):
    envs = create_envs(num=config["num_envs"])

    obs_dim = envs.observation_space.shape[1]
    action_num = envs.action_space[0].n
    agent = PPO(obs_dim, action_num, config, writer=writer)
    # Es braucht viele Episoden is die Policy stabil ist

    rewards = deque(maxlen=config["reward_sliding_window_size"])
    rewards_per_env = {i: [] for i in range(config["num_envs"])}
    intrinsic_rewards_per_env = {i: [] for i in range(config["num_envs"])}

    mean_returns = deque(maxlen=config["return_sliding_window_size"])
    mean_return = 1
    mean_intrinsic_returns = deque(maxlen=config["intrinsic_return_window_size"])
    mean_intrinsic_return = 1
    rewards.appendleft(0)

    epoch = 0
    states, _ = envs.reset()
    hidden_states = torch.zeros((config["num_envs"], config["hidden_size"])).to(agent.device)
    global_step = 0
    while epoch < config["num_epochs"]:
        agent.model.sample_noise()
        for step in range(config["steps_per_env"]):
            global_step += 1 * config["num_envs"]
            action, action_prop, new_h = agent.select_action(states, None, hidden_states, eval=False)
            next_state, reward, terminated, truncated, info = envs.step(action)

            agent_reward = reward 
            agent_reward = (agent_reward / mean_return)
            if config["use_intrinsic_reward"]:
                # Intrinsic rewards
                intrinsic_reward, _ = agent.get_intrinsic_reward(next_state, None, new_h)
                if writer is not None:
                    writer.add_scalar("intrinsic_reward", intrinsic_reward.mean() , global_step)
                agent_reward += intrinsic_reward
            
            for i in range(config["num_envs"]):
                rewards_per_env[i].append(reward[i])
                if config["use_intrinsic_reward"]:
                    intrinsic_rewards_per_env[i].append(intrinsic_reward[i])

                agent.record_obs(states[i], hidden_states[i], action[i], None, agent_reward[i], next_state[i], terminated[i], truncated[i], action_prop[i], True, i, step)
                if info != {}:
                    if terminated[i] or truncated[i]: 
                        agent.memory[i].next_state[step] = torch.FloatTensor(info["final_observation"][i]) / agent.obs_max
                        rewards.appendleft(info["final_info"][i]["episode"]["r"][0])
                        if writer is not None:
                            writer.add_scalar("episodic_return", info["final_info"][i]["episode"]["r"], global_step)
                            writer.add_scalar("episodic_length", info["final_info"][i]["episode"]["l"], global_step)
                        new_h[i] = torch.zeros((1, config["hidden_size"])).to(agent.device)

                        # Calculate return
                        R = 0
                        returns = []
                        for t in reversed(range(len(rewards_per_env[i]))):
                            R = rewards_per_env[i][t] + agent.gamma * R
                            returns.insert(0, R)
                        mean_returns.append(np.mean(np.abs(returns)))

                        # Calculate intrinsic return
                        R = 0
                        returns = []
                        for t in reversed(range(len(intrinsic_rewards_per_env[i]))):
                            R = intrinsic_rewards_per_env[i][t] + agent.gamma * R
                            returns.insert(0, R)
                        mean_intrinsic_returns.append(np.mean(np.abs(returns)))

                        rewards_per_env[i] = []
                        intrinsic_rewards_per_env[i] = []

            states = next_state
            hidden_states = new_h

        if epoch >= config["start_epoch"]:
            agent.model.train()
            agent.train_epochs_bptt(epoch)
            
            # Recaclulate most recent hidden state
            for i in range(config["num_envs"]):
                if not agent.memory[i].terminated[config["steps_per_env"]-1] and not agent.memory[i].truncated[config["steps_per_env"]-1]:
                    state = agent.memory[i].state[config["steps_per_env"]-1].unsqueeze(0).to(agent.device)
                    h_state = agent.memory[i].hidden_state[config["steps_per_env"]-1].unsqueeze(0).to(agent.device)
                    _, _, _, _, new_h = agent.model(state, None, h_state)
                    hidden_states[i] = new_h.detach().cpu()
            agent.memory = {}
        else:
            agent.memory = {}

        if (
            epoch < config["reward_scaling_factor_max_epoch_sampling"]
            and len(mean_returns) > 0
        ):
            mean_return = np.mean(mean_returns) + 1e-8
            mean_intrinsic_return = np.mean(mean_intrinsic_returns) + 1e-8
        
        epoch += 1

        # Testing
        test_env = create_envs(num=1, test=True)
        env_state, _ = test_env.reset()
        ep_reward = 0
        done=False
        state = np.zeros((1, obs_dim))
        state[0] = env_state
        hidden_state = torch.zeros((1, config["hidden_size"])).to(agent.device)
        agent.model.remove_noise()
        agent.model.eval()
        while done == False:
            action, action_prop, hidden_state = agent.select_action(state, None, hidden_state, eval=True)
            env_state, reward, truncated, terminated, info = test_env.step(action[0])
            state[0] = env_state
            ep_reward += reward
            if truncated or terminated: # info["real_done"]:
                break
        #rewards.appendleft(ep_reward)
        print("Epoch " + str(epoch) + "/" + str(config["num_epochs"]) + " Avg. Reward: " + str(sum(rewards)/len(rewards)) + " " + str(ep_reward))
        if writer is not None:
            writer.add_scalar("Avg. train reward", sum(rewards)/len(rewards), epoch)
            writer.add_scalar("Test reward", ep_reward, epoch)
            writer.add_scalar("Mean return", mean_return, epoch)
        
        if epoch % config["model_saving_interval"] == 0:
            agent.save_model(
                "Models/" + config["experiment"] + "-" + config["comment"] + "-agent_" + str(epoch)
            )

        if config["use_obs_max"]:
            agent.update_obs_max()