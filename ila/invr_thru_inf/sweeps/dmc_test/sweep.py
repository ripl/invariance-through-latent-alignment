#!/usr/bin/env python3

from params_proto.neo_hyper import Sweep
from ila.dmc_gen.config import Args
from ila.dmc_gen.config_helper import set_args

import sys; sys.path.append('..')
from const import envs
envs.remove('Reacher-easy')
# envs = ['Cartpole-swingup', 'Ball_in_cup-catch', 'Cheetah-run', 'Cartpole-balance']
envs = ['Cheetah-run']

with Sweep(Args) as sweep:
    with sweep.zip:
        Args.env_name = [f'dmc:{env_name}-v1' for env_name in envs]
        Args.eval_env_name = [f'distracting_control:{env_name}-easy-v1' for env_name in envs]
    with sweep.product:
        Args.seed = [(i + 1) * 100 for i in range(5)]


@sweep.each
def tail(Args):
    # NOTE: pretrained policy (Adapt.snapshot_prefix) does not matter!

    set_args('svea', Args.env_name, Args)

    # snapshot_prefix is only used to generate hash (to be used in get_buffer_prefix).
    # buffer_prefix = str(get_buffer_prefix(BCArgs.env_name, BCArgs.action_repeat, BCArgs.seed, Adapt, eval_env=Args.eval_env_name))[:-4]

    Args.job_name = f"dmc_test/{Args.env_name.split(':', 1)[-1][:-3].lower()}/{Args.seed}"

sweep.save("dmc-test.jsonl")
