#!/usr/bin/env python3

import numpy as np
from params_proto.neo_hyper import Sweep
from iti.dmc_gen.config import Args
from iti.dmc_gen.config_helper import set_args
from invr_thru_inf.config import Adapt, CollectData
import sys; sys.path.append('..')
from const import envs

with Sweep(Args, Adapt, CollectData) as sweep:
    Adapt.latent_buffer_size = 1_000_000
    Adapt.policy_on_clean_buffer = 'random'
    Adapt.policy_on_distr_buffer = 'random'
    CollectData.buffer_type = 'obs_state'

    Adapt.distraction_types = []
    Adapt.distraction_intensity = None

    # eval_env is not used anywhere to avoid confusion
    Args.eval_env_name = None

    with sweep.zip:
        # Args.env_name = [f'dmc:{env_name}-v1' for env_name in envs]
        Args.env_name = [f'dmc:{env_name}-v1' for env_name in envs]

    with sweep.product:
        # Adapt.distraction_types = [['background'], ['camera'], ['video-background'], ['dmcgen-color-hard']]
        Adapt.agent_stem = ['sac', 'svea']
        Args.seed = [(i + 1) * 100 for i in range(5)]


@sweep.each
def tail(Args, Adapt, CollectData):
    # NOTE: pretrained policy (Adapt.snapshot_prefix) does not matter!
    from iti.invr_thru_inf.utils import get_bc_buffer_prefix

    set_args(Args.algorithm, Args.env_name, Args)

    Adapt.snapshot_prefix = f"model-free/model-free/baselines/dmc_gen/run/{Adapt.agent_stem}/{Args.env_name.lower().split(':')[1][:-3]}/{Args.seed}"
    buff_suffix = get_bc_buffer_prefix(Args.env_name, action_repeat=Args.action_repeat, seed=Args.seed,
                                       distraction_types=Adapt.distraction_types, snapshot_prefix=Adapt.snapshot_prefix,
                                       distraction_intensity=Adapt.distraction_intensity, augment=Adapt.augment)
    Args.job_name = f"dc/obs_state/{Adapt.agent_stem}/{buff_suffix}"


sweep.save("obs_state.jsonl")
