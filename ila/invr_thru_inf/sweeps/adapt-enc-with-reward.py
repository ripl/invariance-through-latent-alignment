#!/usr/bin/env python3

import numpy as np
from params_proto.neo_hyper import Sweep
from ila.dmc_gen.config import Args
from ila.dmc_gen.config_helper import set_args
from ila.invr_thru_inf.config import AdaptReward

import sys; sys.path.append('.')
from const import envs
envs = ['Cheetah-run','Cartpole-swingup']

with Sweep(Args, AdaptReward) as sweep:
    AdaptReward.distraction_types = ['background']
    AdaptReward.distraction_intensity = 1.0
    AdaptReward.agent_stem = 'svea'
    AdaptReward.adapt_steps = 100_000

    with sweep.zip:
        Args.env_name = [f'dmc:{env_name}-v1' for env_name in envs]
        Args.eval_env_name = [f'distracting_control:{env_name}-intensity-v1' for env_name in envs]
        AdaptReward.env_name = [f'distracting_control:{env_name}-intensity-v1' for env_name in envs]
        AdaptReward.eval_env_name = [f'distracting_control:{env_name}-intensity-v1' for env_name in envs]

    with sweep.product:
        # AdaptReward.distraction_types = [['background'], ['camera'], ['video-background'], ['dmcgen-color-hard']]
        # AdaptReward.distraction_types = [['background'], ['color'], ['camera']]
        Args.seed = [(i + 1) * 100 for i in range(5)]


@sweep.each
def tail(Args, AdaptReward):
    # NOTE: pretrained policy (AdaptReward.snapshot_prefix) does not matter!
    from ila.invr_thru_inf.utils import get_buffer_prefix
    from const import get_distraction_coef
    AdaptReward.distraction_intensity = AdaptReward.distraction_intensity * get_distraction_coef(AdaptReward.distraction_types) if AdaptReward.distraction_intensity is not None else 0.0

    set_args(Args.algorithm, Args.env_name, Args)

    AdaptReward.snapshot_prefix = f"model-free/model-free/baselines/dmc_gen/run/{AdaptReward.agent_stem}/{Args.env_name.lower().split(':')[1][:-3]}/{Args.seed}"
    buffer_prefix = str(get_buffer_prefix(Args.env_name, Args.action_repeat, Args.seed, AdaptReward, eval_env=Args.eval_env_name))[:-4]
    Args.job_name = f"adapt-reward/sample-bugfix/adapt/{AdaptReward.agent_stem}/{buffer_prefix}/{Args.seed}"

sweep.save("adapt-reward.jsonl")
