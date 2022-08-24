#!/usr/bin/env python3

import numpy as np
from params_proto.neo_hyper import Sweep
from ila.dmc_gen.config import Args
from ila.dmc_gen.config_helper import set_args
from ila.invr_thru_inf.config import Adapt

import sys; sys.path.append('.')
from const import envs

with Sweep(Args, Adapt) as sweep:
    Adapt.latent_buffer_size = 1_000_000
    Adapt.policy_on_clean_buffer = 'random'
    Adapt.policy_on_distr_buffer = 'random'

    # DO NOT use dynamics consistency loss
    Adapt.invm_pretrain_lr = 0
    Adapt.fwdm_pretrain_lr = 0
    Adapt.fwdm_lr = 0
    Adapt.invm_lr = 0
    Adapt.num_invm_steps = 0
    Adapt.invm_kickin_step = 0

    Adapt.gan_lr = 1e-5
    Adapt.gan_disc_lr = 1e-5

    Adapt.augment = True

    # Critical
    Adapt.encoder_from_scratch = False
    Adapt.use_discr_score = False

    Adapt.distraction_types = ['video-background']
    Adapt.distraction_intensity = 1.0

    with sweep.zip:
        Args.env_name = [f'dmc:{env_name}-v1' for env_name in envs]
        # Args.eval_env_name = [f'dmc:{env_name}-v1' for env_name in envs]
        Args.eval_env_name = [f'distracting_control:{env_name}-intensity-v1' for env_name in envs]

    with sweep.product:
        # Adapt.distraction_types = [['background'], ['camera'], ['video-background'], ['dmcgen-color-hard']]
        # Adapt.distraction_types = [['background'], ['color'], ['camera']]

        Adapt.agent_stem = ['sac', 'svea']
        Args.seed = [(i + 1) * 100 for i in range(5)]


@sweep.each
def tail(Args, Adapt):
    # NOTE: pretrained policy (Adapt.snapshot_prefix) does not matter!
    from ila.invr_thru_inf.utils import get_buffer_prefix
    from const import get_distraction_coef
    Adapt.distraction_intensity = Adapt.distraction_intensity * get_distraction_coef(Adapt.distraction_types) if Adapt.distraction_intensity is not None else 0.0

    set_args(Args.algorithm, Args.env_name, Args)

    Adapt.snapshot_prefix = f"model-free/model-free/baselines/dmc_gen/run/{Adapt.agent_stem}/{Args.env_name.lower().split(':')[1][:-3]}/{Args.seed}"
    buffer_prefix = str(get_buffer_prefix(Args.env_name, Args.action_repeat, Args.seed, Adapt, eval_env=Args.eval_env_name))[:-4]
    Args.job_name = f"adpt_video/{Adapt.agent_stem}/{buffer_prefix}/ganlr{Adapt.gan_lr}/gandiscrlr{Adapt.gan_disc_lr}/{Args.seed}"


sweep.save("adpt_video.jsonl")
