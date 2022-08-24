#!/usr/bin/env python3

import numpy as np
from params_proto.neo_hyper import Sweep
from iti.dmc_gen.config import Args
from iti.dmc_gen.config_helper import set_args
from iti.invr_thru_inf.config import Adapt, CollectData
import sys; sys.path.append('..')
from const import envs
envs = ['Cartpole-swingup']
with Sweep(Args, Adapt, CollectData) as sweep:
    Adapt.latent_buffer_size = 1_000_000
    Adapt.policy_on_clean_buffer = 'random'
    Adapt.policy_on_distr_buffer = 'random'
    CollectData.buffer_type = 'target'

    Adapt.augment = False  # !!!

    Adapt.snapshot_prefix = None

    with sweep.zip:
        # Args.env_name = [f'dmc:{env_name}-v1' for env_name in envs]
        Args.env_name = [f'dmc:{env_name}-v1' for env_name in envs]
        Args.eval_env_name = [f'distracting_control:{env_name}-intensity-v1' for env_name in envs]

    with sweep.product:
        # Adapt.distraction_types = [['background'], ['camera'], ['video-background'], ['dmcgen-color-hard']]
        Adapt.distraction_types = [['background'], ['color'], ['camera']]
        Adapt.distraction_intensity = list(np.linspace(0, 1, 5 + 1))[1:]  # remove the first item (0)
        Args.seed = [(i + 1) * 100 for i in range(5)]


@sweep.each
def tail(Args, Adapt, CollectData):
    # NOTE: pretrained policy (Adapt.snapshot_prefix) does not matter!
    from invr_thru_inf.utils import get_buffer_prefix
    from const import get_distraction_coef
    Adapt.distraction_intensity = Adapt.distraction_intensity * get_distraction_coef(Adapt.distraction_types)

    set_args(Args.algorithm, Args.env_name, Args)

    Args.job_name = f"{Adapt.agent_stem}/{get_buffer_prefix(Args.env_name, Args.action_repeat, Args.seed, Adapt, eval_env=Args.eval_env_name)}"


sweep.save("targ_nonaug.jsonl")
