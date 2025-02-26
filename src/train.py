# adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

from dqn import DQN, select_action
from replay_buff import Transition, ReplayMemory

from utils import state_batch_to_tensor
from config import FPS_RATE
import rl_config
from game_gym_env import CarGameEnv

checkpoint_location = "../saves/checkpoints/"
execution_name = "non_supervised"


def optimize_model(policy_net, target_net, optimizer, memory, device):
    if len(memory) < rl_config.BATCH_SIZE:
        return
    transitions = memory.sample(rl_config.BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states_img, non_final_next_states_scr = \
        state_batch_to_tensor([s for s in batch.next_state if s is not None])
    state_img_batch, state_scr_batch = state_batch_to_tensor(batch.state)
    reward_batch = torch.cat(batch.reward)
    action_batch = torch.cat(batch.action)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    q_values = policy_net(state_img_batch, state_scr_batch)
    state_action_values = torch.stack([
        q_values[i].gather(1, action_batch[:, i].unsqueeze(1)).squeeze(1)
        for i in range(len(q_values))
    ], dim=1).sum(dim=1)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(rl_config.BATCH_SIZE, device=device)
    with torch.no_grad():
        next_q_values = target_net(non_final_next_states_img, non_final_next_states_scr)
        next_state_values[non_final_mask] = torch.stack([
            q.max(1).values for q in next_q_values
        ], dim=1).sum(dim=1)
    
    # Compute the expected Q values
    next_state_action_values = (next_state_values * rl_config.GAMMA) + reward_batch

    # Compute Huber loss, square loss if small but linear if big
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, next_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=rl_config.LR, amsgrad=True, weight_decay=rl_config.L2_LOSS_WEIGHT)
    memory = ReplayMemory(rl_config.REPLAY_BUFF_SIZE)

    steps_done = 0

    env = CarGameEnv()

    num_episodes = rl_config.NUM_EPISODES

    scores = np.zeros(num_episodes)

    for i_episode in range(1, num_episodes+1):
        # Initialize the environment and get its state
        print("=========episode %d========" % i_episode)
        state, info = env.reset()
        # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            if t % FPS_RATE == 0:
                print(t//FPS_RATE)
            eps_threshold = rl_config.EPS_END + (rl_config.EPS_START - rl_config.EPS_END) * \
                np.exp(-1. * steps_done / rl_config.EPS_DECAY)
            steps_done += 1
            action = select_action(env, policy_net, state, eps_threshold)
            observation, reward, terminated, truncated, _ = env.step(action)
            reward = torch.tensor([reward], device=device)
            scores[i_episode-1] += reward
            done = terminated or truncated
            if terminated:
                next_state = None
            else:
                next_state = observation
                # torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state,
                        torch.tensor(action, device=device).unsqueeze(0),
                        next_state,
                        reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(policy_net, target_net, optimizer, memory, device)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = \
                    policy_net_state_dict[key]*rl_config.TAU \
                    + target_net_state_dict[key]*(1 - rl_config.TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                print("reward of ep %d: %d" % (i_episode, scores[i_episode - 1]))
                break

        if i_episode % rl_config.TORCH_CHECKPOINT_FREQ == 0:
            torch.save({
                "episode": i_episode,
                "reward": scores[i_episode-1],
                "policy_net_dict": policy_net.state_dict(),
                "target_net_dict": target_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, "%s%s_%d.pt" % (checkpoint_location, execution_name, i_episode))
            with open("../saves/games/%s_%d.pkl" % (execution_name, i_episode), "wb") as f:
                pickle.dump(memory.memory, f)
            np.save("../saves/scores/%s_scores.npy" % execution_name, scores)

    env.close()

    # plt.plot(scores)
    # plt.show()


if __name__ == "__main__":
    main()
