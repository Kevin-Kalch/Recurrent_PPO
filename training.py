import cProfile
from collections import deque
import pstats
import gymnasium
import torch
from model import PPO
import numpy as np
torch.manual_seed(4020)
from gymnasium.wrappers import TimeAwareObservation
from torch.utils.tensorboard import SummaryWriter
#import flappy_bird_gymnasium

def create_envs(num=4):
    envs = []
    for i in range(num):
        def gen():
            env = LastAction(VelHidden(gymnasium.make('LunarLander-v2')))
            #env = LastAction(gymnasium.make('FlappyBird-v0', use_lidar=True))
            env = gymnasium.wrappers.RecordEpisodeStatistics(env)
            return env
        envs.append(gen)
    envs = gymnasium.vector.SyncVectorEnv(envs)
    return envs

profiler = cProfile.Profile()

def main():
    num_envs = 16
    steps_per_env = 128
    num_epochs = 1000

    envs = create_envs(num=num_envs)
    obs_dim = envs.observation_space.shape[1]
    action_num = envs.action_space[0].n
    writer = SummaryWriter(comment="intrinsic_loss_low_dont_std")
    #writer = None
    agent = PPO(obs_dim, action_num, steps_per_env, writer=writer)
    # Es braucht viele Episoden is die Policy stabil ist

    rewards = deque(maxlen=32)
    rewards_per_env = {i: [] for i in range(num_envs)}#
    intrinsic_rewards_per_env = {i: [] for i in range(num_envs)}
    mean_returns = deque(maxlen=4096)
    mean_return = 1
    std_intrinsic_returns = deque(maxlen=4096)
    std_intrinsic_return = 1
    rewards.appendleft(0)

    epoch = 0
    states, _ = envs.reset()
    hidden_states = torch.zeros((num_envs, 64)).to(agent.device)
    global_step = 0
    while epoch < num_epochs:
        # if epoch == 5:
        #     profiler.enable()
        agent.model.sample_noise()
        agent.model.eval()
        for step in range(steps_per_env):
            global_step += 1 * num_envs
            action, action_prop, new_h = agent.select_action(states, None, hidden_states, eval=False)
            next_state, reward, terminated, truncated, info = envs.step(action)

            # Intrinsic rewards
            intrinsic_reward, _ = agent.get_intrinsic_reward(next_state, None, new_h)
            if writer is not None:
                writer.add_scalar("charts/intrinsic_reward", (intrinsic_reward / std_intrinsic_return).mean() , global_step)

            agent_reward = reward 
            agent_reward = (agent_reward / mean_return) + intrinsic_reward
            
            for i in range(num_envs):
                rewards_per_env[i].append(reward[i])
                intrinsic_rewards_per_env[i].append(intrinsic_reward[i])
                agent.record_obs(states[i], hidden_states[i], action[i], None, agent_reward[i], next_state[i], terminated[i], truncated[i], action_prop[i], i, step)
                if info != {}:
                    if terminated[i] or truncated[i]: 
                        agent.memory[i].next_state[step] = torch.FloatTensor(info["final_observation"][i]) / agent.obs_max
                        rewards.appendleft(info["final_info"][i]["episode"]["r"])
                        if writer is not None:
                            writer.add_scalar("charts/episodic_return", info["final_info"][i]["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["final_info"][i]["episode"]["l"], global_step)
                        new_h[i] = torch.zeros((1, 64)).to(agent.device)

                        # Calculate return
                        R = 0
                        returns = []
                        for t in reversed(range(len(rewards_per_env[i]))):
                            R = rewards_per_env[i][t] + agent.gamma * R
                            returns.insert(0, R)
                        mean_returns.append(np.mean(np.abs(returns)))

                        R = 0
                        returns = []
                        for t in reversed(range(len(intrinsic_rewards_per_env[i]))):
                            R = intrinsic_rewards_per_env[i][t] + agent.gamma * R
                            returns.insert(0, R)
                        std_intrinsic_returns.append(np.std(returns))

                        rewards_per_env[i] = []
                        intrinsic_rewards_per_env[i] = []

            states = next_state
            hidden_states = new_h

        if epoch >= 5:
            agent.model.train()
            agent.train_epochs_bptt_2(epoch)
            
            # Recaclulate most recent hidden state
            for i in range(num_envs):
                if not agent.memory[i].terminated[steps_per_env-1] and not agent.memory[i].truncated[steps_per_env-1]:
                    state = agent.memory[i].state[steps_per_env-1].unsqueeze(0).to(agent.device)
                    h_state = agent.memory[i].hidden_state[steps_per_env-1].unsqueeze(0).to(agent.device)
                    _, _, _, _, new_h = agent.model(state, None, h_state)
                    hidden_states[i] = new_h.detach().cpu()
            agent.memory = {}
        else:
            agent.memory = {}

        if epoch < 100:
            mean_return = np.mean(mean_returns) + 1e-8
            std_intrinsic_return = np.mean(std_intrinsic_returns) + 1e-8
        print(np.mean(mean_return))
        
        epoch += 1

        test_env = LastAction(VelHidden(gymnasium.make('LunarLander-v2')))
        state, _ = test_env.reset()
        ep_reward = 0
        done=False
        hidden_state = torch.zeros((1, 64))
        agent.model.remove_noise()
        while done == False:
             action, action_prop, hidden_state = agent.select_action(state, None, hidden_state, eval=True)
             state, reward, truncated, terminated, info = test_env.step(action)
             ep_reward += reward
             if truncated or terminated: # info["real_done"]:
                 break
        #rewards.appendleft(ep_reward)
        print("Epoch " + str(epoch) + "/" + str(num_epochs) + " Avg. Reward: " + str(sum(rewards)/len(rewards)) + " " + str(ep_reward))
        if writer is not None:
            writer.add_scalar("Avg. train Reward", sum(rewards)/len(rewards), epoch)
            writer.add_scalar("Test Reward", ep_reward, epoch)
            writer.add_scalar("Mean Var", mean_return, epoch)
        
        if epoch % 100 == 0:
            agent.save_model("LunarLander-agent_" + str(epoch))



class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        self.update_from_moments(batch_mean, batch_var, len(x))

    def update_from_moments(self, batch_mean, batch_var, batch_count):        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / (tot_count - 1)
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class EpisodicLifeEnv(gymnasium.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.

    :param env: Environment to wrap
    """

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        self.lives = lives
        info['real_done'] = self.was_real_done
        return obs, reward, terminated, truncated, info

class VelHidden(gymnasium.ObservationWrapper):
    def observation(self, obs):
        obs[[2,3]] = 0.0
        return obs
    
class OutsideViewport(gymnasium.Wrapper):
    def step(self, action):
        next_state, reward, terminated, truncated, info = super().step(action)
        if next_state[1] > 1.5:
            terminated = True
            reward = -100
        if next_state[0] < -1.5 or next_state[0] > 1.5:
            terminated = True
            reward = -100
        return next_state, reward, terminated, truncated, info

class LastAction(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Box(
            low=np.append(self.observation_space.low, np.zeros(self.action_space.n)), 
            high=np.append(self.observation_space.high, np.ones(self.action_space.n)), 
            shape=(env.observation_space.shape[0] + self.action_space.n,), 
            dtype=np.float32)

    def step(self, action):
        next_state, reward, terminated, truncated, info = super().step(action)
        actions = np.zeros(self.action_space.n)
        actions[action] = 1
        next_state = np.append(next_state, actions)
        return next_state, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        state, info = super().reset(seed=seed, options=options)
        state = np.append(state, np.zeros(self.action_space.n))
        return state, info


if __name__ == '__main__':
    #torch.autograd.set_detect_anomaly(True)
    torch.set_num_threads(64)
    torch.set_float32_matmul_precision('high')
    torch.set_printoptions(sci_mode=False)
    torch.backends.cudnn.benchmark = True
    
    main()
    # profiler.disable()
    # with open("output.txt", "w+") as f:
    #     stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
    #     stats = stats.strip_dirs()
    #     stats.sort_stats('cumtime').print_stats()