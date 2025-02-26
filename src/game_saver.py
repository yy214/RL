import pygame
from collections import namedtuple, deque
import pickle
from os import path

from utils import get_rl_state

from racing_game import Game
from config import *

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

GAME_NAME = "chaotic"

def get_state(game):
    return {
        "image":get_rl_state(game.carViewWindow, game.lowresImg, game.car),
        "speed":game.car.vel,
        # "timeLeft": TIME_LIMIT-self.game.timer,
        "currLap":(game.score+1) // len(game.checkpoints),
        "boostsLeft":game.car.boostCount
    }

def main():
    transitionsOfGame = deque()

    game = Game()
    game.reset()

    isAccelerating = False
    isBreaking = False
    steerDirection = 1
    tryBoosting = False

    gameStarted = False # press to start
    done = False

    state = get_state(game)

    while not done:
        if not gameStarted:
            if pygame.event.peek(pygame.KEYDOWN) \
            and pygame.key.get_pressed()[pygame.K_UP]:
                gameStarted = True
        if not gameStarted:
            game.display()
            continue

        prevState = state
        # Process player inputs.
        tryBoosting = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Get mouse position
                mouse_pos = pygame.mouse.get_pos()
                print(f"Mouse clicked at: {mouse_pos}")
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT: 
                    steerDirection += 1
                if event.key == pygame.K_RIGHT:
                    steerDirection -= 1
                if event.key == pygame.K_UP:
                    isAccelerating = True
                if event.key == pygame.K_DOWN:
                    isBreaking = True
                if event.key == pygame.K_SPACE:
                    tryBoosting = True
                if event.key == pygame.K_PLUS:
                    game.car.BASE_ACCEL += 0.01
                    print(game.car.BASE_ACCEL)
                if event.key == pygame.K_MINUS:
                    game.car.BASE_ACCEL -= 0.01
                    print(game.car.BASE_ACCEL)
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT: 
                    steerDirection -= 1
                if event.key == pygame.K_RIGHT:
                    steerDirection += 1
                if event.key == pygame.K_UP:
                    isAccelerating = False
                if event.key == pygame.K_DOWN:
                    isBreaking = False
        reward = game.inputProcessing(steerDirection, isAccelerating, isBreaking, tryBoosting)
        game.display(DEBUG=True)
        game.timeUpdate(framerate=FPS_RATE)

        state = get_state(game)

        if game.score >= len(game.checkpoints) * LAP_COUNT:
            reward += TIME_LIMIT - game.timer
            transitionsOfGame.append(Transition(prevState, [steerDirection, isAccelerating, isBreaking, tryBoosting], state, reward))
            pygame.quit()
            break
        
        transitionsOfGame.append(Transition(prevState, [steerDirection, isAccelerating, isBreaking, tryBoosting], state, reward))
        if game.timer >= TIME_LIMIT:
            pygame.quit()
            break

    count = 1
    while 1:
        fileChecked = "../saves/games/%s_%d.pkl" % (GAME_NAME, count)
        if not path.exists(fileChecked):
            with open(fileChecked, "wb") as f:
                pickle.dump(transitionsOfGame, f)
            break
        count += 1


if __name__ == "__main__":
    main()