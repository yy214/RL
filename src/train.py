import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import state_batch_to_tensor
from config import *
from rl_config import *
from game_gym_env import CarGameEnv

checkpoint_location = "../saves/checkpoints/"

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # need to sample differently, see prioritized experience replay Schaul et al. 2016
        return np.random.choice(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.fc_size = 64*3*1
        self.fc_image = nn.Linear(self.fc_size, 128)  # Flatten image output
        self.fc_scalar = nn.Linear(3, 32)  # 3 scalar features
        self.fc_merge = nn.Linear(128 + 32, 256)
        self.fc_out = nn.Linear(256, sum([3, 2, 2, 2]))  # Output Q-values for each discrete action

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, image, scalar):
        # image: 1*30*15
        x = self.pool(F.relu(self.conv1(image)))  # 16*30*15 -> 16*15*7
        x = self.pool(F.relu(self.conv2(x)))  # 32*15*7 -> 32*7*3
        x = self.pool(F.relu(self.conv3(x)))  # 64*7*3 -> 64*3*1
        x = x.view(-1, self.fc_size)
        x = F.relu(self.fc_image(x))

        s = F.relu(self.fc_scalar(scalar))
        merged = torch.cat((x, s), dim=1)
        merged = F.relu(self.fc_merge(merged))
        q_values = self.fc_out(merged)
        return torch.split(q_values, [3, 2, 2, 2], dim=1)


def select_action(env, policy_net, state, eps_threshold):
    # epsilon-greedy approach
    sample = np.random.uniform()
    if sample > eps_threshold:
        image, scalar = state_batch_to_tensor([state])
        q_values = policy_net(image, scalar)
        with torch.no_grad():
            # Take highest Q-value per action
            action = [torch.argmax(q, dim=1).detach().cpu().numpy()[0] for q in q_values]
            return action
            # return policy_net(state).max(1).indices.view(1, 1)
    else:
        return env.action_space.sample()


def optimize_model(policy_net, target_net, optimizer, memory, device):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states_img, non_final_next_states_scr = state_batch_to_tensor([s for s in batch.next_state
                                                if s is not None])
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
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_q_values = target_net(non_final_next_states_img, non_final_next_states_scr)
        next_state_values[non_final_mask] = torch.stack([
            q.max(1).values for q in next_q_values
        ], dim=1).sum(dim=1)
    
    # Compute the expected Q values
    next_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
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

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True, weight_decay=L2_LOSS_WEIGHT)
    memory = ReplayMemory(REPLAY_BUFF_SIZE)

    steps_done = 0

    env = CarGameEnv()

    num_episodes = NUM_EPISODES

    scores = np.zeros(num_episodes)

    for i_episode in range(1,num_episodes+1):
        # Initialize the environment and get its state
        print("=========episode %d========"%i_episode)
        state, info = env.reset()
        # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                np.exp(-1. * steps_done / EPS_DECAY)
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
                        torch.tensor([action]),
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
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
            
            if done:
                break
        
        if i_episode % TORCH_CHECKPOINT_FREQ == 0:
            torch.save({
                "episode": i_episode,
                "reward": scores[i_episode-1],
                "policy_net_dict": policy_net.state_dict(), 
                "target_net_dict": target_net.state_dict(), 
                "optimizer_state_dict": optimizer.state_dict(), 
            }, "%scheckpoint_%d.pt" % (checkpoint_location, i_episode))

    env.close()
    np.save("../saves/scores/scores.npy", scores)

    # plt.plot(scores)
    # plt.show()
    # plt.savefig("../saves/scores/scores.png")


if __name__ == "__main__":
    main()