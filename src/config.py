import numpy as np

FPS_RATE = 60
TIME_LIMIT = FPS_RATE * 120  # 2 min

COLOR_BLACK = (0, 0, 0)

MARIO_DISPLAY_SIZE = (750, 773)
FOV_RATIO = 10
FOV_SIZE = (30, 15)
POV_POS = (1, 7)
DISPLAY_SIZE = (MARIO_DISPLAY_SIZE[0] + 310, MARIO_DISPLAY_SIZE[1] + 100)  # width, height 
CAR_SIZE = 20

CHECKPOINT_POS = [((-10, 330), (120, 330)),
                  ((177, 0), (169, 151)),
                  ((743, 199), (590, 205)), 
                  ((733, 571), (584, 561)), 
                  ((146, 756), (190, 612))]
LAP_COUNT = 3
INIT_POS = np.array([60, 346], dtype=np.float64)


CAR_STEERING = 4
CAR_BASE_ACCEL = 0.05
CAR_BOOST_ACCEL = 0.2
CAR_BASE_BRAKE = 0.1
CAR_ROAD_FRICTION = 0.003
CAR_OFF_ROAD_FRICTION = 0.02
CAR_INIT_SPEED_BOOST_COUNT = 3
CAR_BOOST_DURATION = 30
CAR_TOP_SPEED = 10 # prevents RL bugs