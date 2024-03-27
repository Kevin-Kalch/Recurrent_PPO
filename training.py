from collections import deque
import gymnasium
import torch
from model import PPO
import numpy as np
torch.manual_seed(4020)

def create_envs(num=4):
    envs = []
    for i in range(num):
        env = VelHidden(gymnasium.make('LunarLander-v2'))
        envs.append(env)
    return envs

#profiler = cProfile.Profile()

def main():
    num_envs = 8
    steps_per_env = 128
    num_epochs = 5000

    envs = create_envs(num=num_envs)
    obs_dim = envs[0].observation_space.shape[0]
    action_num = envs[0].action_space.n
    env_infos = [
        {"state":None,"env":env, "ep_reward":0, "env_id": i, "reward_memory": [], "hidden_state": None}
        for i, env in enumerate(envs)
    ]
    agent = PPO(obs_dim, action_num, steps_per_env)

    collect_steps = steps_per_env*num_envs  
    rewards = deque(maxlen=32)
    vars = []
    mean_vars = 1
    rewards.appendleft(0)

    epoch = 0
    while epoch < num_epochs:
        # if epoch == 1:
        #     profiler.enable()
        exp_collected = 0
        obs = []
        while exp_collected < collect_steps:
            for env_info in env_infos:
                if env_info["state"] is None:
                    env_info["state"], _ = env_info["env"].reset()
                    env_info["ep_reward"] = 0

                
                action, action_prop, new_h = agent.select_action(env_info["state"], env_info["hidden_state"])
                next_state, reward, terminated, truncated, info = env_info["env"].step(action)
                agent_reward = reward
                if truncated:
                  agent_reward -= mean_vars

                agent_reward = agent_reward / mean_vars
                agent.record_obs(env_info["state"], env_info["hidden_state"], action, agent_reward, next_state, terminated, truncated, action_prop, env_info["env_id"], exp_collected // len(env_infos))
                env_info["ep_reward"] += reward 
                env_info["reward_memory"].append(reward)
                
                env_info["state"] = next_state
                env_info["hidden_state"] = new_h
                exp_collected += 1
                
                if truncated or terminated:
                    if True: # info["real_done"]:
                        R = 0
                        returns = []
                        for r in reversed(env_info["reward_memory"]):
                            R = r + 0.99 * R
                            returns.insert(0, R)
                        vars.append(np.std(returns))
                        
                        env_info["state"] = None
                        rewards.appendleft(env_info["ep_reward"])
                        env_info["reward_memory"] = []
                        env_info["hidden_state"] = None

                    obs.append(next_state)

                if exp_collected > collect_steps:
                    break
            if exp_collected > collect_steps:
                    break
        
        if epoch >= 5:
            agent.train_epochs_bptt_3()
            
            # Recaclulate most recent hidden state
            for env_info in env_infos:
                if env_info["hidden_state"] is not None:
                    state = agent.memory[env_info["env_id"]].state[-1].unsqueeze(0).to(agent.device)
                    h_state = agent.memory[env_info["env_id"]].hidden_state[-1].unsqueeze(0).to(agent.device)
                    _, _, new_h = agent.model(state, h_state)
                    env_info["hidden_state"] = new_h.detach().cpu()
            agent.memory = {}
        else:
            agent.memory = {}

        if len(vars) > 3 and len(vars) < 128:
            mean_vars = np.max(vars) + 1e-8
        print(np.mean(mean_vars))
        
        epoch += 1

        test_env = create_envs(num=1)[0]
        state, _ = test_env.reset()
        ep_reward = 0
        done=False
        hidden_state = None
        while done == False:
             action, action_prop, hidden_state = agent.select_action(state, hidden_state)
             next_state, reward, truncated, terminated, info = test_env.step(action)
             state = next_state
             ep_reward += reward
             if truncated or terminated:
                 break
        print("Epoch " + str(epoch) + "/" + str(num_epochs) + " Avg. Reward: " + str(sum(rewards)/len(rewards)) + " " + str(ep_reward))

        if epoch % 10 == 0:
            agent.save_model("SpaceInvaders-v5-agent_" + str(epoch))

class VelHidden(gymnasium.ObservationWrapper):
    def observation(self, obs):
        obs[[2,3,5]] = 0.0
        return obs


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    main()