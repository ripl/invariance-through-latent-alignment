#!/usr/bin/env python3

from params_proto.neo_hyper import Sweep
from ila.dmc_gen.config import Args
from ila.invr_thru_inf.config import Adapt, CollectData
import sys; sys.path.append('..')
from const import envs

with Sweep(Args, Adapt, CollectData) as sweep:
    Adapt.latent_buffer_size = 1_000_000
    Adapt.policy_on_clean_buffer = 'random'
    Adapt.policy_on_distr_buffer = 'random'
    Adapt.use_sample_bugfix = True
    CollectData.buffer_type = 'original'

    Args.eval_env_name = None
    Adapt.augment = False

    with sweep.zip:
        # Args.env_name = [f'dmc:{env_name}-v1' for env_name in envs]
        Args.env_name = [f'dmc:{env_name.lower()}-v1' for env_name in envs]

    with sweep.product:
        Adapt.agent_stem = ['sac', 'soda', 'svea', 'pad']
        Args.seed = [(i + 1) * 100 for i in range(5)]


@sweep.each
def tail(Args, Adapt, CollectData):
    # NOTE: pretrained policy (Adapt.snapshot_prefix) does not matter!
    from invr_thru_inf.utils import get_buffer_prefix

    if Args.env_name == 'dmc:Finger-spin-v1':
        Args.action_repeat = 2
    elif Args.env_name == 'dmc:Cartpole-swingup-v1':
        Args.action_repeat = 8
    else:
        Args.action_repeat = 4  # default

    Adapt.snapshot_prefix = f"model-free/model-free/baselines/dmc_gen/run/{Adapt.agent_stem}/{Args.env_name.lower().split(':')[1][:-3]}/{Args.seed}"
    Args.job_name = f"sample-bugfix/{Adapt.agent_stem}/{get_buffer_prefix(Args.env_name, Args.action_repeat, Args.seed, Adapt, latent_buffer=True)}"
    # RUN.job_name = f"{Adapt.agent_stem}/{get_buffer_prefix(Args.env_name, Args.action_repeat, Args.seed, Adapt, eval_env=Args.eval_env_name)}"


sweep.save("orig.jsonl")
