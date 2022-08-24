#!/usr/bin/env python3

soda_envs = ['Walker-walk', 'Walker-stand', 'Cartpole-swingup', 'Ball_in_cup-catch', 'Finger-spin']
extra_envs = ['Reacher-easy', 'Cheetah-run', 'Cartpole-balance', 'Finger-turn_easy']

envs = soda_envs + extra_envs
time_limit = 60 * 60 * 24.3

def get_snapshot_prefix(algorithm, env_name, seed, debug=False):
    if algorithm == 'drqv2':
        return f'model-free/model-free/baselines/drqv2_original/train/{env_name}-v1/{seed}'
    elif algorithm in ['soda', 'sac', 'pad', 'svea']:
        if debug:
            return f'model-free/model-free/baselines/dmc_gen/train/test---dmc_gen_nockpt/{algorithm}/{env_name}-v1/{seed}'
        # return f'model-free/model-free/baselines/dmc_gen/train/dmc_gen_nockpt/{algorithm}/{env_name}-v1/{seed}'
        return f'model-free/model-free/baselines/dmc_gen/train/{algorithm}/{env_name}-v1/{seed}'
    else:
        raise ValueError(f'algorithm {algorithm} is invalid')


def get_distraction_coef(distraction_types):
    if distraction_types == ['camera']:
        return 0.25  # interporate between 0 and 0.25
    elif distraction_types == ['color']:
        return 0.5  # interporate between 0 and 0.5
    elif distraction_types == ['background']:
        return 1.0
    elif distraction_types == ['video-background']:
        return 1.0
    elif distraction_types == []:
        return 0.0
    else:
        raise ValueError(f'distraction_type {distraction_types} is invalid')
