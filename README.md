# My reinforcement learning project for APM_53670_EP
General usage:
- requirements: numpy, pytorch, pygame, gymnasium. Using `conda` instead of `pip` to install pygame may result in issues on some Linux based distributions.
- find all the code in the "src/" folder. Make sure to run any code while being in the "src/" folder or the relative paths won't work properly.
- you may find the game parameters in "src/config.py" (game physics, display, etc.)
- find the hyperparameters for the RL part in rl_config.py 

## Training a model

## Evaluating a trained model
In test_trained_model.py, modify the `checkpoint_location` to the wanted pytorch checkpoint to test it.

During evaluation, any input (keyboard or mouse or tab change) messes up the pygame render for some reason, so avoid them.

If you need to quit the pygame window, you'll have to quit twice, the 2nd time is intended to crash the pygame client.