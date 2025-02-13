import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from car_physics import Car
from checkpoint import Checkpoint
from racing_game import Game

from utils import get_rl_state

from config import *

class CarGameEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CarGameEnv, self).__init__()

        self.state = None
        self.action_space = spaces.MultiDiscrete([3, 2, 2, 2]) # steer (-1, 0, 1), accel, brake, boost

        image = spaces.Box(low=0, high=255, shape=FOV_SIZE, dtype=np.uint8)
        speed = spaces.Box(low=0.0, high=1e3, shape=(), dtype=np.float32)
        timeLeft = spaces.Box(low=0, high=TIME_LIMIT, dtype=np.uint16)
        boostsLeft = spaces.Discrete(CAR_INIT_SPEED_BOOST_COUNT+1)

        self.observation_space = spaces.Dict({
            "image":image,
            "speed":speed,
            "timeLeft":timeLeft,
            "boostsLeft":boostsLeft,
        })
        self.game = Game()
    
    def _get_state(self):
        return get_rl_state(self.game.carViewWindow, self.game.lowresImg, self.game.car)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self._get_state(), {}

    def step(self, action):
        # steerDirection, isAccelerating, isBreaking, tryBoosting = action
        reward = self.game.inputProcessing(*action)
        # if rendermode = human then self.game.display(DEBUG=True)
        self.game.timeUpdate()

        truncated = self.game.timer >= TIME_LIMIT

        terminated = False
        if self.game.score >= len(self.game.checkpoints) * LAP_COUNT:
            reward += TIME_LIMIT - self.game.timer
            terminated = True

        return self._get_state(), reward, terminated, truncated, {}
    
    def render(self, mode="human"):
        # self.game.display(DEBUG=True)
        pass

    def close(self):
        # Cleanup when closing the environment (optional)
        pygame.quit()
        pass