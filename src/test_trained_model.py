from game_gym_env import CarGameEnv
from dqn import DQN, select_action
import torch
from config import FPS_RATE

checkpoint_location = "../saves/checkpoints/pretrain_n_RL_150.pt"

def main():
    env = CarGameEnv(render_mode="human")
    policy_net = DQN()
    state_dict = torch.load(checkpoint_location, map_location=torch.device("cpu"))
    policy_net.load_state_dict(state_dict["policy_net_dict"])
    policy_net.eval()

    state, _ = env.reset()

    done = False
    t = 0
    while not done:
        t += 1
        if t % FPS_RATE == 0:
            print(t // FPS_RATE)
        action = select_action(env, policy_net, state, 0, device="cpu")
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()
        
    env.close()

main()