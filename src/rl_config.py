NUM_EPISODES = 600
TORCH_CHECKPOINT_FREQ = 10

# ===== HYPERPARAMETERS ======
BATCH_SIZE = 32  # try 128?
GAMMA = 0.99
LR = 1e-4
REPLAY_BUFF_SIZE = 1000*1000

# epsilon-greedy
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

# demonstrations
PRETRAINING_STEPS = 10*1000
DEMO_BONUS = 5e-3  # same as in https://github.com/hw9603/DQfD-PyTorch/
DIFF_MOVE_PENALTY = 0.1  # for supervised loss

# combining losses
N_STEP_LOSS_WEIGHT = 1
EXPERT_LOSS_WEIGHT = 1
L2_LOSS_WEIGHT = 1e-3
