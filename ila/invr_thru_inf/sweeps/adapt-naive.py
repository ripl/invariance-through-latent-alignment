#!/usr/bin/env python3

import numpy as np
from params_proto.neo_hyper import Sweep
from ila.dmc_gen.config import Args
from ila.dmc_gen.config_helper import set_args
from ila.invr_thru_inf.config import NaiveAdapt

import sys; sys.path.append('.')
from const import envs
# envs = ['Walker-walk']
# envs = ['Cartpole-swingup']
envs = ['Walker-walk', 'Walker-stand', 'Ball_in_cup-catch', 'Finger-spin']
envs += ['Cheetah-run', 'Cartpole-balance']

with Sweep(Args, NaiveAdapt) as sweep:
    NaiveAdapt.latent_buffer_size = 1_000_000
    NaiveAdapt.policy_on_clean_buffer = 'random'
    NaiveAdapt.policy_on_distr_buffer = 'random'

    # NaiveAdapt.augment = True
    NaiveAdapt.augment = False

    # Critical
    NaiveAdapt.encoder_from_scratch = False

    NaiveAdapt.distraction_types = ['background']
    NaiveAdapt.distraction_intensity = 1.0

    NaiveAdapt.agent_stem = 'svea'

    NaiveAdapt.num_adapt_steps = 100_000

    NaiveAdapt.adapt_eval_every_steps = 20
    NaiveAdapt.num_shuffled_losses = 0

    with sweep.zip:
        Args.env_name = [f'dmc:{env_name}-v1' for env_name in envs]
        # Args.eval_env_name = [f'dmc:{env_name}-v1' for env_name in envs]
        Args.eval_env_name = [f'distracting_control:{env_name}-intensity-v1' for env_name in envs]

    with sweep.product:
        # NaiveAdapt.distraction_types = [['background'], ['camera'], ['video-background'], ['dmcgen-color-hard']]
        # NaiveAdapt.distraction_types = [['background'], ['color'], ['camera']]
        Args.seed = [(i + 1) * 100 for i in range(5)]


@sweep.each
def tail(Args, NaiveAdapt):
    # NOTE: pretrained policy (NaiveAdapt.snapshot_prefix) does not matter!
    from ila.invr_thru_inf.utils import get_buffer_prefix
    from const import get_distraction_coef
    NaiveAdapt.distraction_intensity = NaiveAdapt.distraction_intensity * get_distraction_coef(NaiveAdapt.distraction_types) if NaiveAdapt.distraction_intensity is not None else 0.0

    set_args(Args.algorithm, Args.env_name, Args)

    NaiveAdapt.snapshot_prefix = f"model-free/model-free/baselines/dmc_gen/run/{NaiveAdapt.agent_stem}/{Args.env_name.lower().split(':')[1][:-3]}/{Args.seed}"
    buffer_prefix = str(get_buffer_prefix(Args.env_name, Args.action_repeat, Args.seed, NaiveAdapt, eval_env=Args.eval_env_name))[:-4]
    Args.job_name = f"adapt-naive/sample-bugfix/adapt/{NaiveAdapt.agent_stem}/{buffer_prefix}/{Args.seed}"


sweep.save("adapt-naive.jsonl")
