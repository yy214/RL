import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from utils import state_batch_to_tensor


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


def evaluate_q_a(q, a):
    return torch.stack([
        q[i].gather(1, a[:, i].unsqueeze(1)).squeeze(1)
        for i in range(len(q))
    ], dim=1).sum(dim=1)


def get_q_a_values(net, state_img_batch, state_scr_batch, action_batch=None):
    # calculates Q(s,a)
    return evaluate_q_a(net(state_img_batch, state_scr_batch), action_batch)


def get_q_values(net, state_img_batch, state_scr_batch):
    # calculate max Q(s)
    q_values = net(state_img_batch, state_scr_batch)
    return torch.stack([
        q.max(1).values for q in q_values
    ], dim=1).sum(dim=1)
