# greatly adapted from https://github.com/hw9603/DQfD-PyTorch/

import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dqn import DQN, select_action, get_q_a_values, get_q_values, evaluate_q_a
from replay_buff import Transition, Memory

from utils import state_batch_to_tensor
from config import FPS_RATE, TIME_LIMIT
import rl_config
from game_gym_env import CarGameEnv

savegame_location = "../saves/games/"
checkpoint_location = "../saves/checkpoints/"


def loadDemonstrations():
    TODO: check struct of replays


def storeDemoTransition(s, a, r, s_, done, demoEpisode):
    s = torch.Tensor(s) FIX HERE
    s_ = torch.Tensor(s_) FIX HERE
    episodeReplay = self.demoReplay[demoEpisode]  # replay of certain demo episode
    index = len(episodeReplay) WHY
    data = (s, a, r, s_, done, (demoEpisode, index))
    episodeReplay.append(data)
    replay.add(data)

def storeTransition(self, s, a, r, s_, done):
    s = torch.Tensor(s) FIX HERE
    s_ = torch.Tensor(s_) FIX HERE
    replay.add((s, a, r, s_, done, None))


def getQValues(batch, policy_net, target_net, device):
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*batch)) #CHECK HERE
    
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states_img, non_final_next_states_scr = \
        state_batch_to_tensor([s for s in batch.next_state if s is not None])
    state_img_batch, state_scr_batch = state_batch_to_tensor(batch.state)
    reward_batch = torch.cat(batch.reward)
    action_batch = torch.cat(batch.action)

    state_action_values = get_q_a_values(policy_net, state_img_batch, state_scr_batch, action_batch)
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(rl_config.BATCH_SIZE, device=device)

    # calcQ
    with torch.no_grad():
        next_state_values[non_final_mask] = \
            get_q_values(target_net, non_final_next_states_img, non_final_next_states_scr)
    
    # Compute the expected Q values
    next_state_action_values = (next_state_values * rl_config.GAMMA) + reward_batch

    return state_action_values, next_state_action_values

def getSupervisedQ(q_values, aE):
    a1 = [torch.argmax(q_values, dim=1).detach().cpu().numpy()[0] for q in q_values]
    if (a1==aE).all():
        a_seconds = None
        differences = q_values[a1] - q_values[a_seconds] #FIX HERE
        ind = argmax of differences
        a1[ind] = a_seconds[ind]
    return evaluate_q_a(q_values, a1)

def calcSupervisedLoss(batch, policy_net):
    loss = torch.tensor(0.0)
    count = 0  # number of demo
    for s, aE, *_, isdemo in batch:
        if isdemo is None:
            continue
        img, scr = state_batch_to_tensor([s])
        q_values = policy_net(img, scr)
        QE = evaluate_q_a(q_values, aE)  # aE may be incorrect in format
        A1, A2 = np.array(A)[:2]  # action with largest and second largest Q
        maxA = A2 if (A1 == aE).all() else A1
        Q = get_q_a_values(policy_net, img, scr, maxA)
        if (Q + rl_config.DIFF_MOVE_PENALTY) < QE:
            continue
        else:
            loss += (Q - QE)[0]
            count += 1
    return loss / count if count != 0 else loss

def calcNStepLoss():
    return 0 # TODO

# CHECK
def optimizeModel(optimizer, replay_buff, policy_net, target_net):
    batch, idxs, weights = replay_buff.sample(rl_config.BATCH_SIZE)
    state_action_values, next_state_action_values = \
        getQValues(batch, policy_net, target_net, device)

    for i in range(rl_config.BATCH_SIZE):
        td_err = state_action_values[i] - next_state_action_values[i]
        replay_buff.update(idxs[i], abs(td_err))

    td_loss = (weights * F.smooth_l1_loss(state_action_values, next_state_action_values, reduction="none")).mean()
    total_loss = td_loss \
        + rl_config.EXPERT_LOSS_WEIGHT*calcSupervisedLoss() \
        + rl_config.N_STEP_LOSS_WEIGHT*calcNStepLoss()

    optimizer.zero_grad()
    total_loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    # return to check if the pretraining can be stopped early
    return total_loss 

def updateTargetNet(policy_net, target_net):
    target_net.load_state_dict(policy_net.state_dict)
    target_net.eval()

def pretrainingLoop():
    TARGET_UPDATE_FREQ = 1000
    loss = 0
    for i in range(1, rl_config.PRETRAINING_STEPS+1):
        loss += optimizeModel()
        if i % TARGET_UPDATE_FREQ == 0:
            print("pretraining %d/%d: %.2f" 
                  % (i, rl_config.PRETRAINING_STEPS, loss//TARGET_UPDATE_FREQ))
            # TODO: if loss small then no need to continue training

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
    
    memory = Memory(rl_config.BATCH_SIZE)
    demoReplay = loadDemonstrations()

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

            if done:
                print("reward of ep %d: %d" % (i_episode, scores[i_episode - 1]))
                break
        
        updateTargetNet(policy_net, target_net)

        if i_episode % rl_config.TORCH_CHECKPOINT_FREQ == 0:
            torch.save({
                "episode": i_episode,
                "reward": scores[i_episode-1],
                "policy_net_dict": policy_net.state_dict(),
                # "target_net_dict": target_net.state_dict(), # policy_net = target_net at the end of episode
                "optimizer_state_dict": optimizer.state_dict(),
            }, "%scheckpoint_%d.pt" % (checkpoint_location, i_episode))
            np.save("../saves/scores/scores.npy", scores)
            
            TODO: save replay buffer

    env.close()

if __name__ == "__main__":
    main()
