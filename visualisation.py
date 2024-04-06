import time
import gymnasium
from model import PPO
import numpy as np
from gymnasium.wrappers import TimeAwareObservation

def main():
    #env = gymnasium.make('ALE/SpaceInvaders-v5', obs_type="ram", render_mode="human")
    env = TimeAwareObservation(VelHidden(gymnasium.make('LunarLander-v2', render_mode="human")))
    obs_dim = env.observation_space.shape[0]
    action_num = env.action_space.n
    obs_dim = 6
    agent = PPO(obs_dim, action_num, 1)
    agent.load_model("SpaceInvaders-v5-agent_1000")
    while True:
        state, _ = env.reset()
        env.render()
        ep_reward = 0
        done=False
        h = None
        while done == False:
            action, _, h = agent.select_action(state, h)
            next_state, reward, done, terminated, _ = env.step(action)
            ep_reward += reward
            if done or terminated:
                h = None
                break
            state = next_state
            #env.render()
            time.sleep(0.001)

class VelHidden(gymnasium.ObservationWrapper):
    def observation(self, obs):
        obs[[2,3,5]] = 0.0
        return obs[[0,1,4,6,7]]

if __name__ == '__main__':
    main()
