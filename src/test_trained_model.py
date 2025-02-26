from game_gym_env import CarGameEnv
from dqn import DQN, select_action
import torch
# from copy import deepcopy

checkpoint_location = "../saves/checkpoints/"

def main():
    env = CarGameEnv(render_mode="human")
    policy_net = DQN()
    state_dict = torch.load(checkpoint_location)
    policy_net.load_state_dict(state_dict)
    policy_net.eval()

    state, _ = env.reset()

    done = False
    while not done:
        action = select_action(env, policy_net, state, 0)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()
    env.close()

main()