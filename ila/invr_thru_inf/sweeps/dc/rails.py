#!/usr/bin/env python3

import numpy as np
from params_proto.neo_hyper import Sweep
from iti.dmc_gen.config import Args
from iti.dmc_gen.config_helper import set_args
from invr_thru_inf.config import Adapt, CollectData
import sys; sys.path.append('..')
# from const import envs
envs = ['Cartpole-swingup', 'Ball_in_cup-catch', 'Cheetah-run', 'Cartpole-balance']

with Sweep(Args, Adapt, CollectData) as sweep:
    Adapt.latent_buffer_size = 1_000_000
    Adapt.policy_on_clean_buffer = 'random'
    Adapt.policy_on_distr_buffer = 'random'

    CollectData.buffer_type = 'obs_rails'

    Adapt.distraction_intensity = None

    with sweep.zip:
        # Args.env_name = [f'dmc:{env_name}-v1' for env_name in envs]
        Args.env_name = [f'dmc:{env_name}-v1' for env_name in envs]
        # Args.eval_env_name = [f'dmc:{env_name}-v1' for env_name in envs]
        # NOTE: We don't use intensity!!!
        Args.eval_env_name = [f'distracting_control:{env_name}-easy-v1' for env_name in envs]

    with sweep.product:
        # Adapt.distraction_types = [['background'], ['camera'], ['video-background'], ['dmcgen-color-hard']]
        Adapt.distraction_types = [['background'], ['color'], ['camera'], ['background', 'color', 'camera']]
        Adapt.agent_stem = ['sac', 'svea']
        Args.seed = [(i + 1) * 100 for i in range(5)]


@sweep.each
def tail(Args, Adapt, CollectData):
    # NOTE: pretrained policy (Adapt.snapshot_prefix) does not matter!
    from invr_thru_inf.utils import get_buffer_prefix

    set_args(Args.algorithm, Args.env_name, Args)

    Adapt.snapshot_prefix = f"model-free/model-free/baselines/dmc_gen/run/{Adapt.agent_stem}/{Args.env_name.lower().split(':')[1][:-3]}/{Args.seed}"
    Args.job_name = f"dc/obs_rails/{Adapt.agent_stem}/{get_buffer_prefix(Args.env_name, Args.action_repeat, Args.seed, Adapt, eval_env=Args.eval_env_name)}"


sweep.save("obs_rails.jsonl")
