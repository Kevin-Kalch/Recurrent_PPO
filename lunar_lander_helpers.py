import gymnasium
import numpy as np

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