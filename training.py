from collections import deque
import gymnasium
import torch
from model import PPO
import numpy as np
torch.manual_seed(4020)
from gymnasium.wrappers import TimeAwareObservation
from torch.utils.tensorboard import SummaryWriter

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
    num_epochs = 300

    envs = create_envs(num=num_envs)
    obs_dim = envs[0].observation_space.shape[0]
    action_num = envs[0].action_space.n
    env_infos = [
        {"state":None,"env":env, "ep_reward":0, "env_id": i, "reward_memory": [], "hidden_state": None}
        for i, env in enumerate(envs)
    ]
    writer = SummaryWriter(comment="")
    #writer = None
    agent = PPO(obs_dim, action_num, steps_per_env, writer=writer)
    # Es braucht viele Episoden is die Policy stabil ist

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
        while exp_collected < collect_steps:
            for env_info in env_infos:
                if env_info["state"] is None:
                    env_info["state"], _ = env_info["env"].reset()
                    env_info["ep_reward"] = 0

                action, action_prop, new_h = agent.select_action(env_info["state"], env_info["hidden_state"])
                next_state, reward, terminated, truncated, info = env_info["env"].step(action)
                agent_reward = reward
                if truncated:
                    print("Truncated")
                    agent_reward -= mean_vars

                agent_reward = agent_reward / mean_vars
                if truncated:
                    agent_reward += agent.model.get_value(torch.from_numpy(next_state).clone().float().unsqueeze(0).to(agent.device), new_h)[0].item()
                agent.record_obs(env_info["state"], env_info["hidden_state"], action, agent_reward, next_state, terminated, truncated, action_prop, env_info["env_id"], exp_collected // len(env_infos))
                env_info["ep_reward"] += reward 
                env_info["reward_memory"].append(reward)
                
                env_info["state"] = next_state
                env_info["hidden_state"] = new_h
                exp_collected += 1
                
                if truncated or terminated:
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

                if exp_collected >= collect_steps:
                    break
            if exp_collected >= collect_steps:
                    break
        if epoch >= 5:
            agent.train_epochs_bptt(epoch)
            
            # Recaclulate most recent hidden state
            for env_info in env_infos:
                if env_info["hidden_state"] is not None:
                    state = agent.memory[env_info["env_id"]].state[steps_per_env-1].unsqueeze(0).to(agent.device)
                    h_state = agent.memory[env_info["env_id"]].hidden_state[steps_per_env-1].unsqueeze(0).to(agent.device)
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
             action, action_prop, hidden_state = agent.select_action(state, hidden_state, eval=True)
             state, reward, truncated, terminated, info = test_env.step(action)
             ep_reward += reward
             if truncated or terminated: # info["real_done"]:
                 break
        #rewards.appendleft(ep_reward)
        print("Epoch " + str(epoch) + "/" + str(num_epochs) + " Avg. Reward: " + str(sum(rewards)/len(rewards)) + " " + str(ep_reward))
        if writer is not None:
            writer.add_scalar("Avg. train Reward", sum(rewards)/len(rewards), epoch)
            writer.add_scalar("Test Reward", ep_reward, epoch)
            writer.add_scalar("Mean Var", mean_vars, epoch)
        
        if epoch % 100 == 0:
            agent.save_model("SpaceInvaders-v5-agent_" + str(epoch))



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

from gymnasium import spaces

class TestEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, render_mode=None, id=0):
        self.id = id
        self.size = 5  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.state_step = 0
        self.observation_space = spaces.Box(0, 1000, shape=(1,), dtype=int)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        self.state_step = 0
        return np.array([int(str(self.id) + "0" + str(self.state_step))]), {}
    
    def step(self, action):
        self.state_step += 1
        if self.state_step == 1000:
            return np.array([int(str(self.id) + "0" + str(self.state_step))]), int(self.state_step%2), True, False, {}
        return np.array([int(str(self.id) + "0" + str(self.state_step))]), int(self.state_step%2), False, False, {}

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.set_printoptions(sci_mode=False)
    
    main()
    # profiler.disable()
    # with open("output.txt", "w+") as f:
    #     stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
    #     stats = stats.strip_dirs()
    #     stats.sort_stats('cumtime').print_stats()