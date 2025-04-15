import time
import gymnasium
from model import PPO
import numpy as np
import torch
from main import config
from lunar_lander_helpers import VelHidden, LastAction
from gymnasium.wrappers import TimeAwareObservation

def main():
    env = TimeAwareObservation(LastAction(LastAction(VelHidden(gymnasium.make('LunarLander-v3', render_mode="human")))))
    obs_dim = env.observation_space.shape[0]
    action_num = env.action_space.n
    agent = PPO(obs_dim, action_num, config, None)
    agent.load_model("Models/LunarLander-v3-PPO Test, no recalc-agent_1250")
    agent.model.remove_noise()
    agent.model.eval()

    while True:
        env_state, _ = env.reset()
        env.render()
        ep_reward = 0
        done=False
        state = np.array([env_state])
        h = torch.zeros((1, agent.config["hidden_size"])).to(agent.device)
        while done is False:
            action, _, h = agent.select_action(state, None, h, eval=True)
            next_state, reward, done, terminated, _ = env.step(action[0])
            ep_reward += reward
            if done or terminated:
                h = None
                break
            state[0] = next_state
            env.render()
            time.sleep(0.001)

if __name__ == '__main__':
    main()
