import time
import gymnasium
from model import PPO
import numpy as np
import torch
from main import config
from lunar_lander_helpers import VelHidden, LastAction

def main():
    env = LastAction(VelHidden(gymnasium.make('LunarLander-v2', render_mode="human")))
    obs_dim = env.observation_space.shape[0]
    action_num = env.action_space.n
    agent = PPO(obs_dim, action_num, config, None)
    agent.load_model("Models/LunarLander-v2-GePPO Test, v-trace, geppo vtrace, geppo adv, no es, no lr adjust, not recalc returns, fixed probs in adv, changed expl, less epochs, tanh-agent_225")
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
