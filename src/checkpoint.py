import pygame
import numpy as np
import copy

from config import CHECKPOINT_POS
from car_physics import Car

class Checkpoint:
    def __init__(self, e1, e2):
        r"""A checkpoint for counting laps (and to avoid ultra shortcuts)"""
        self.e1 = copy.deepcopy(e1) # on the left
        self.e2 = copy.deepcopy(e2)
        self.seg = (self.e2[0] - self.e1[0], self.e2[1] - self.e1[1])

    def isPassed(self, car: Car):
        carSegment = car.pos - car.prevPos
        return (np.cross(self.seg, car.pos - self.e1) <= 0
                and np.cross(self.seg, car.prevPos - self.e1) >= 0
                and np.cross(carSegment, self.e1 - car.prevPos) <= 0
                and np.cross(carSegment, self.e2 - car.prevPos) >= 0)
    
    def draw(self, win, color, width):
        pygame.draw.line(win, color, self.e1, self.e2, width)
        pygame.draw.circle(win, color, self.e2, width)

def generateCheckpoints():
    checkpoints = []
    for (e1, e2) in CHECKPOINT_POS:
        checkpoints.append(Checkpoint(e1, e2))
    return checkpoints

def debugShowCheckpoints(checkpoints, currCheckPoint, screen):
    for i, checkPt in enumerate(checkpoints):
        if i != currCheckPoint:
            checkPt.draw(screen, (0,255,0), 2)
        else:
            checkPt.draw(screen, (0,0,255), 2)