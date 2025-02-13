import pygame
import numpy as np

from config import COLOR_BLACK, MARIO_DISPLAY_SIZE, FOV_RATIO, POV_POS

def blitRotateCenter(surf, image, pos, angle=0):
    """
    Rotate a surface and blit it to the window
    :param surf: the surface to blit to
    :param image: the image surface to rotate
    :param pos: the center position of the image
    :param angle: a float value for angle
    :return: None
    """
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(center=pos).center)

    surf.blit(rotated_image, new_rect.topleft)

def get_rl_state(carViewWindow, lowresImg, car):
    carViewWindow.fill(COLOR_BLACK)
    rotatedMap = pygame.transform.rotate(lowresImg, -car.angle)
    rotatedMapRect = rotatedMap.get_rect()
    # center-relative positions
    dx = (car.pos[0] - MARIO_DISPLAY_SIZE[0]//2) // FOV_RATIO
    dy = (car.pos[1] - MARIO_DISPLAY_SIZE[1]//2) // FOV_RATIO
    
    angleRad = car.angle * np.pi / 180 # Negative because Pygame rotates clockwise
    rotatedDx = dx * np.cos(angleRad) - dy * np.sin(angleRad)
    rotatedDy = dx * np.sin(angleRad) + dy * np.cos(angleRad)

    # Translate back to top-left-relative coordinates of the rotated image
    rotatedX = rotatedMapRect.centerx + rotatedDx
    rotatedY = rotatedMapRect.centery + rotatedDy

    carViewWindow.blit(rotatedMap, (POV_POS[0] - rotatedX, POV_POS[1] - rotatedY))
    return pygame.surfarray.array_red(carViewWindow)
    # later: input_tensor = torch.tensor(surface_array, dtype=torch.float32).unsqueeze(0) #[1, height, width]
    # input_tensor = input_tensor.unsqueeze(0) #[1, 1, height, width]