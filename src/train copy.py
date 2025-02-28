# greatly adapted from https://github.com/hw9603/DQfD-PyTorch/ + https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import os
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict as ddict
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dqn import DQN, select_action, get_q_a_values, get_q_values, evaluate_q_a
from replay_buff import Transition, TransitionDemo, Memory

from utils import state_batch_to_tensor
from config import FPS_RATE, TIME_LIMIT
import rl_config
from game_gym_env import CarGameEnv

savegame_location = "../saves/games/"
ignored_savegames = ["chaotic", "savegame"]
replay_buff_save_location = "../saves/replay_buffs/"
checkpoint_location = "../saves/checkpoints/"
NEED_PRETRAIN = False  # if you need the pretrain part
NO_RL_TRAIN = False  # if you only want the pretrain part
starting_state_dict = "../saves/checkpoints/pretrain_n_step.pt"  # if not None, loads a state_dict as initialization. Useful if you have a pretrained model
starting_ep = 0
execution_name = "pretrain_n_step_RL"  # name given to the saved files

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


def storeDemoTransition(s, a, s_, r, memory, demoReplay, demoEpisode):
    if r > 1:  # hardcoded: detects if done
        s_ = None
    episodeReplay = demoReplay[demoEpisode]  # for n-step
    index = len(episodeReplay)
    data = TransitionDemo(s,
                          torch.tensor(a, device=device).unsqueeze(0),
                          s_,
                          torch.tensor([r], device=device),
                          (demoEpisode, index))
    episodeReplay.append(data)
    memory.add(data)


def storeTransition(s, a, s_, r, done, memory):
    if done:
        s_ = None
    memory.add(TransitionDemo(s,
                              torch.tensor(a, device=device).unsqueeze(0),
                              s_,
                              torch.tensor([r], device=device),
                              None))


def contains_any(s, substrings):
    return any(sub in s for sub in substrings)


def loadDemonstrations(memory, demoReplay):
    ep = 0
    for entry in os.listdir(savegame_location):
        full_path = os.path.join(savegame_location, entry)
        if not os.path.isfile(full_path) or contains_any(full_path, ignored_savegames):
            continue
        with open(full_path, "rb") as file:
            print(f"Reading {full_path}")
            loaded = pickle.load(file)  # deque
            for t in loaded:
                storeDemoTransition(t.state,
                                    t.action,
                                    t.next_state,
                                    t.reward,
                                    memory,
                                    demoReplay,
                                    ep)
        ep += 1


def getQValues(batch, policy_net, target_net):
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = TransitionDemo(*zip(*batch))

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
    if len(non_final_next_states_img.shape) == 4:  # prevents incidents if all are final states
        with torch.no_grad():
            next_state_values[non_final_mask] = \
                get_q_values(target_net, non_final_next_states_img, non_final_next_states_scr)

    # Compute the expected Q values
    next_state_action_values = (next_state_values * rl_config.GAMMA) + reward_batch

    return state_action_values, next_state_action_values


def getSupervisedQ(q_values, aE):
    best_actions = []
    second_best_actions = []
    min_diff = 1e42
    ind = 0

    for i, sub_q in enumerate(q_values):
        top2 = torch.topk(sub_q, 2)  # Get top 2 values and indices
        best_actions.append(top2.indices[0, 0].item())  # Best action index
        second_best_actions.append(top2.indices[0, 1].item())
        diff = top2.values[0, 0].item() - top2.values[0, 1].item()
        if min_diff > diff:
            min_diff = diff
            ind = i

    a1 = torch.tensor(best_actions, device=device).unsqueeze(0)
    if torch.equal(a1, aE):
        a2 = best_actions[:]
        a2[ind] = second_best_actions[ind]
        a2 = torch.tensor(a2, device=device).unsqueeze(0)
        return evaluate_q_a(q_values, a2)
    return evaluate_q_a(q_values, a1)


def calcSupervisedLoss(batch, policy_net):
    loss = torch.tensor(0.0, device=device)
    count = 0  # number of demo
    for elem in batch:
        if elem.isDemo is None:
            continue
        img, scr = state_batch_to_tensor([elem.state])
        aE = elem.action
        q_values = policy_net(img, scr)
        QE = evaluate_q_a(q_values, aE)
        Q = getSupervisedQ(q_values, aE)
        if (Q + rl_config.DIFF_MOVE_PENALTY) < QE:
            continue
        else:
            loss += (Q - QE)[0]
            count += 1
    return loss / count if count != 0 else loss


def calcNStepLoss(batch, state_action_values, target_net, demoReplay):
    r"""only for demonstrations, TODO refactor for generic"""
    loss = torch.tensor(0.0, device=device)
    count = 0
    for i, elem in enumerate(batch):
        if elem.isDemo is None:
            continue
        episode, idx = elem.isDemo
        nidx = idx + rl_config.N_STEP
        lepoch = len(demoReplay[episode])
        count += 1
        if nidx >= lepoch:
            discountedR = torch.tensor(0.0, device=device)
        else:
            b = demoReplay[episode][nidx]
            if b.next_state:
                # print(episode, nidx, lepoch, b.next_state)
                img, scr = state_batch_to_tensor([b.next_state])
                discountedR = get_q_values(target_net, img, scr)
            else:
                discountedR = torch.tensor(0.0, device=device)
        for j in range(min(nidx, lepoch)-1, idx-1, -1):
            b = demoReplay[episode][j]
            discountedR = b.reward + rl_config.GAMMA * discountedR

        predict = state_action_values[i]
        loss += (discountedR - predict)[0] ** 2
    return loss / count


def optimize_model(policy_net, target_net, optimizer, replay_buff, demoReplay):
    batch, idxs, weights = replay_buff.sample(rl_config.BATCH_SIZE)
    weights = torch.tensor(weights).to(device)
    state_action_values, next_state_action_values = \
        getQValues(batch, policy_net, target_net)

    for i in range(rl_config.BATCH_SIZE):
        td_err = state_action_values[i] - next_state_action_values[i]
        replay_buff.update(idxs[i], abs(td_err))

    # print(F.smooth_l1_loss(state_action_values,
    #                        next_state_action_values,
    #                        reduction="none").shape, weights.shape)
    td_loss = (weights
               * F.smooth_l1_loss(state_action_values,
                                  next_state_action_values,
                                  reduction="none")).mean()
    total_loss = td_loss \
        + rl_config.EXPERT_LOSS_WEIGHT*calcSupervisedLoss(batch, policy_net) \
        + rl_config.N_STEP_LOSS_WEIGHT*calcNStepLoss(batch, state_action_values, target_net, demoReplay)

    optimizer.zero_grad()
    total_loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    # return to check if the pretraining can be stopped early
    return total_loss


def updateTargetNet(policy_net, target_net):
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()


def pretrainingLoop(nb_steps, policy_net, target_net, optimizer, memory, demoReplay):
    TARGET_UPDATE_FREQ = 100
    loss = 0
    for i in range(1, nb_steps+1):
        loss += optimize_model(policy_net, target_net, optimizer, memory, demoReplay)
        if i % TARGET_UPDATE_FREQ == 0:
            print("pretraining %d/%d: %.2f"
                  % (i, nb_steps, loss//TARGET_UPDATE_FREQ))
            updateTargetNet(policy_net, target_net)
            loss = 0
            # TODO? if loss small then no need to continue training


def main():
    policy_net = DQN().to(device)
    optimizer = optim.AdamW(policy_net.parameters(),
                            lr=rl_config.LR,
                            amsgrad=True,
                            weight_decay=rl_config.L2_LOSS_WEIGHT)
    if starting_state_dict:
        s_dict = torch.load(starting_state_dict)
        policy_net.load_state_dict(s_dict["policy_net_dict"])
        policy_net.eval()
        optimizer.load_state_dict(s_dict["optimizer_state_dict"])
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())

    demoReplay = ddict(list)
    memory = Memory(rl_config.BATCH_SIZE)
    loadDemonstrations(memory, demoReplay)

    if NEED_PRETRAIN:
        pretrainingLoop(rl_config.PRETRAINING_STEPS,
                        policy_net,
                        target_net,
                        optimizer,
                        memory,
                        demoReplay)

    if NO_RL_TRAIN:  # check pretraining
        torch.save({
            "policy_net_dict": policy_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, "%s%s.pt" % (checkpoint_location, execution_name))
        return

    steps_done = 0

    env = CarGameEnv()

    num_episodes = rl_config.NUM_EPISODES

    scores = np.zeros(num_episodes)

    for i_episode in range(starting_ep + 1, num_episodes+1):
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
            next_state, reward, terminated, truncated, _ = env.step(action)
            scores[i_episode-1] += reward
            done = terminated or truncated

            # Store the transition in memory
            storeTransition(state, action, next_state, reward, done, memory)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(policy_net, target_net, optimizer, memory, demoReplay)

            if done:
                print("reward of ep %d: %d" % (i_episode, scores[i_episode - 1]))
                break

        updateTargetNet(policy_net, target_net)

        if i_episode % rl_config.TORCH_CHECKPOINT_FREQ == 0:
            torch.save({
                "episode": i_episode,
                "policy_net_dict": policy_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, "%s%s_%d.pt" % (checkpoint_location, execution_name, i_episode))
            with open("%s%s_%d.pkl" % (replay_buff_save_location, execution_name, i_episode), "wb") as f:
                pickle.dump(memory.tree, f)
            np.save("../saves/scores/%s_scores.npy" % execution_name, scores)

    env.close()


if __name__ == "__main__":
    main()
