import pygame
import numpy as np
from utils import blitRotateCenter
import copy

from config import *
    
class Car:
    def __init__(self, 
                 img, 
                 pos:np.ndarray, 
                 angle,
                 debugMode = False
                 ):
        """Initialize the object
        :pos: starting pos (np.ndarray, 2D)
        :return: None
        """
        assert pos.shape == (2,)
        self.prevPos = copy.deepcopy(pos)
        self.pos = copy.deepcopy(pos)
        self.angle = angle
        self.vel = 0
        self.boostCount = CAR_INIT_SPEED_BOOST_COUNT
        self.boostLeft = CAR_BOOST_DURATION//3 #start with tiny boost

        self.img = img
        self.rotatedImg = pygame.transform.rotate(self.img, self.angle)
        self.rotatedRect = self.rotatedImg.get_rect(center = self.img.get_rect(center = self.pos).center)
        self.rotatedMask = pygame.mask.from_surface(self.rotatedImg)
        
        self.debugMode = debugMode
    
    def updateMask(self):
        self.rotatedImg = pygame.transform.rotate(self.img, self.angle)
        self.rotatedRect = self.rotatedImg.get_rect(center = self.img.get_rect(center = self.pos).center)
        self.rotatedMask = pygame.mask.from_surface(self.rotatedImg)

    def steer(self, orientation:int):
        r"""Steer the car. The amount of steering depends on the speed
        :param orientation: 0 for right and +2 for left
        """
        orientation -= 1
        assert abs(orientation) == 1, "The value should be 0 or 2"
        self.angle += orientation * CAR_STEERING / (1 + self.vel)

    def accelerate(self):
        if self.boostLeft == 0: # acceleration is overriden by boost
            self.vel += CAR_BASE_ACCEL #TODO: max speed / less accel when high speed?

    def boost(self):
        if self.boostCount > 0:
            self.boostCount -= 1
            self.boostLeft = CAR_BOOST_DURATION

    def brake(self):
        self.vel = max(self.vel - CAR_BASE_BRAKE, 0)

    def speedDecay(self, isOffroad):
        r"""A friction physic"""
        if not isOffroad:
            self.vel -= CAR_ROAD_FRICTION*(self.vel**2)
        else:
            self.vel -= CAR_OFF_ROAD_FRICTION*(self.vel**2)

    def move(self, isOffroad):
        if self.boostLeft > 0:
            self.vel += CAR_BOOST_ACCEL
            self.boostLeft -= 1
        self.speedDecay(isOffroad)
        self.vel = min(CAR_TOP_SPEED, self.vel)
        self.updateMask()
        self.prevPos = copy.deepcopy(self.pos)
        self.pos += self.vel * np.array([np.cos(self.angle * np.pi / 180), -np.sin(self.angle * np.pi / 180)])


    def getCollision(self, otherMask): # assumes otherMask has topleft at 0,0
        return otherMask.overlap(self.rotatedMask, (self.rotatedRect.left, self.rotatedRect.top)) 

    def getCollisionMask(self, otherMask):
        return self.rotatedMask.overlap_mask(otherMask, (-self.rotatedRect.left, -self.rotatedRect.top))

    def wallCollisionHandle(self, wallMask):
        collisionMask = self.getCollisionMask(wallMask)
        centroid = collisionMask.centroid()
        if centroid == (0,0): return # no collision

        carMaskSize = self.rotatedMask.get_size()
        axis = int(abs(carMaskSize[0]//2 - centroid[0]) < abs(carMaskSize[1]//2 - centroid[1]))

        sign = 1 if carMaskSize[axis]//2 - centroid[axis] > 0 else -1
        self.pos[axis] += sign * 2 * (carMaskSize[axis]//2 - abs(carMaskSize[axis]//2 - centroid[axis]))
        self.vel *= 0.5

    def draw(self, win):
        """Draw the car
        :param win: pygame window or surface
        :return: None
        """
        blitRotateCenter(win, self.img, self.pos, self.angle)
        if self.debugMode:
            pygame.draw.circle(win, (255,255,255), self.pos, 4)
            pygame.draw.circle(win, (255,0,255), self.prevPos, 4)