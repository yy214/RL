import pygame
import numpy as np
from car_physics import Car
from checkpoint import generateCheckpoints, debugShowCheckpoints
from utils import get_rl_state

from config import *

class Game:
    def __init__(self):
        self.checkpoints = generateCheckpoints()

        pygame.init()

        self.screen = pygame.display.set_mode(DISPLAY_SIZE)
        self.racetrackImg = pygame.image.load("../assets/racetrack.png")

        self.offroadImg = pygame.image.load("../assets/offroad.png").convert_alpha()
        self.offroadMask = pygame.mask.from_surface(self.offroadImg, 0)

        self.collisionsImg = pygame.image.load("../assets/walls.png").convert_alpha()
        self.collisionMask = pygame.mask.from_surface(self.collisionsImg, 0)

        self.carViewWindow = pygame.Surface(FOV_SIZE)
        self.lowresImg = pygame.image.load("../assets/lowres.png").convert_alpha()

        pygame.display.set_caption("Car game")

        self.carImg = pygame.image.load("../assets/car.png").convert_alpha()
        self.carImg = pygame.transform.rotate(self.carImg, -90)
        self.carImg = pygame.transform.scale(self.carImg, (CAR_SIZE,CAR_SIZE))

        # for displays
        pygame.font.init()
        self.font = pygame.font.SysFont("comicsans", 25)
        self.gameStartFont = pygame.font.SysFont("comicsans", 50)
        self.clock = pygame.time.Clock() #check compatibility with AI

    def reset(self):
        self.car = Car(self.carImg, INIT_POS, 90, True)
        # input processing -> ????
        self.isAccelerating = False
        self.isBreaking = False
        self.steerDirection = 1 

        self.score = 0 # curr checkpoint
        self.timer = 0

    def inputProcessing(self, steerDirection, isAccelerating, isBreaking, tryBoosting):
        if tryBoosting: self.car.boost()
        if isAccelerating: self.car.accelerate()
        if isBreaking: self.car.brake()
        if steerDirection != 1: self.car.steer(steerDirection)

        self.car.move(self.car.getCollision(self.offroadMask))
        self.car.wallCollisionHandle(self.collisionMask)

        self.currCheckPoint = (1 + self.score) % len(self.checkpoints)
        if self.checkpoints[self.currCheckPoint].isPassed(self.car):
            self.score += 1
            return 1
        return 0

    def display(self, DEBUG=False):
        self.screen.fill(COLOR_BLACK)
        self.screen.blit(self.racetrackImg, (0, 0))
        self.screen.blit(self.collisionsImg, (0, 0))
        self.screen.blit(self.offroadImg, (0, 0))
        
        if DEBUG:
            debugShowCheckpoints(self.checkpoints, self.currCheckPoint, self.screen)
            stateView = get_rl_state(self.carViewWindow, self.lowresImg, self.car)
            debugScaledView = pygame.transform.scale_by(self.carViewWindow, FOV_RATIO)
            self.screen.blit(debugScaledView, (MARIO_DISPLAY_SIZE[0] + 10, 10))

        self.car.draw(self.screen)
        # boosts left
        boostCountDisplay = self.font.render(f"Boosts Remaining: {self.car.boostCount}", True, (255,255,255))
        self.screen.blit(boostCountDisplay, (10, MARIO_DISPLAY_SIZE[1]+10))
        # current lap
        currentLap = self.score // len(self.checkpoints)
        lapDisplay = self.font.render(f"Lap {currentLap}/{LAP_COUNT}", True, (255, 255, 255))
        self.screen.blit(lapDisplay, (10, MARIO_DISPLAY_SIZE[1]+60))
        # timer display
        t = self.timer
        t, timerMins = t%(60*FPS_RATE), t//(60*FPS_RATE)
        t, timerSec = t%FPS_RATE, t//FPS_RATE
        timeDisplay = self.font.render(f"Time: {timerMins:02}'{timerSec:02}\"{int(100*t/FPS_RATE):02}", True, (255,255,255))
        self.screen.blit(timeDisplay, (DISPLAY_SIZE[0]//2, MARIO_DISPLAY_SIZE[1]+10))

        pygame.display.update()

    def timeUpdate(self, framerate=None):
        self.timer += 1
        if framerate:
            self.clock.tick(framerate)