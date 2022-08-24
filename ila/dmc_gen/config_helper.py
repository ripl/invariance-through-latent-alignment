#!/usr/bin/env python3

def set_args(algorithm, env_name, Args):
    # Image size
    if algorithm == 'soda':
        Args.image_size = 100
    elif algorithm == 'pad':
        Args.image_size = 100
    else:
        Args.image_size = 84

    # num_layers
    if algorithm == 'pad':
        Args.num_shared_layers = 8
        Args.num_head_layers = 3
    else:
        Args.num_shared_layers = 11
        Args.num_head_layers = 0

    # action repeat
    if "Finger-spin" in env_name:
        Args.action_repeat = 2
    elif "Cartpole-swingup" in env_name:
        Args.action_repeat = 8
    else:
        Args.action_repeat = 4  # default

    # learning rate
    if "Cheetah-run" in env_name:
        # NOTE: This only shows up on PAD paper,
        # but given that other algorithms are not tested on Cheetah
        # and shares the codebase, I apply this across algorithms.
        Args.aux_lr = 3e-4
        Args.actor_lr = 3e-4
        Args.critic_lr = 3e-4
    elif algorithm == 'soda':
        Args.aux_lr = 3e-4
        Args.actor_lr = 1e-3
        Args.critic_lr = 1e-3
    else:
        Args.aux_lr = 1e-3
        Args.actor_lr = 1e-3
        Args.critic_lr = 1e-3
