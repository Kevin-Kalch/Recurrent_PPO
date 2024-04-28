import time
import gymnasium
from model import PPO
import numpy as np
from gymnasium.wrappers import TimeAwareObservation

def main():
    #env = gymnasium.make('ALE/SpaceInvaders-v5', obs_type="ram", render_mode="human")
    env = LastAction(VelHidden(gymnasium.make('LunarLander-v2', render_mode="human")))
    obs_dim = env.observation_space.shape[0]
    action_num = env.action_space.n
    obs_dim = 12
    agent = PPO(obs_dim, action_num, 1)
    agent.load_model("SpaceInvaders-v5-agent_3800")
    while True:
        state, _ = env.reset()
        env.render()
        ep_reward = 0
        done=False
        h = None
        while done == False:
            action, _, h = agent.select_action(state, h, eval=False)
            next_state, reward, done, terminated, _ = env.step(action[0])
            ep_reward += reward
            if done or terminated:
                h = None
                break
            state = next_state
            #env.render()
            time.sleep(0.001)

class VelHidden(gymnasium.ObservationWrapper):
    def observation(self, obs):
        obs[[2,3]] = 0.0
        return obs

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
    main()
