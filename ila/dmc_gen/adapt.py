#!/usr/bin/env python3
import os
from os.path import join as pjoin
import torch

from iti.invr_thru_inf.adapt import Progress
import wandb
from iti.helpers import logger, mllogger


def load_adpt_agent(Args, Adapt):
    """
    Load from the pretrained agent specified by snapshot_prefix.
    """
    from iti.invr_thru_inf.env_helpers import get_env
    from iti.invr_thru_inf.agent import AdaptationAgent

    dummy_env = get_env(Args.env_name, Args.frame_stack, Args.action_repeat, Args.seed,
                        size=Args.image_size)
    action_shape = dummy_env.action_space.shape
    obs_shape = dummy_env.observation_space.shape

    from iti.invr_thru_inf.dummy_actors import DMCGENDummyActor
    snapshot_dir = pjoin(Args.checkpoint_root, Adapt.snapshot_prefix)
    snapshot_path = pjoin(snapshot_dir, 'actor_state.pt')
    logger.print('Loading model from', snapshot_path)
    actor_state = mllogger.load_torch(path=snapshot_path)

    from iti.dmc_gen.algorithms.factory import make_agent
    mock_agent = make_agent(obs_shape, action_shape, Args)
    mock_agent.actor.load_state_dict(actor_state)
    encoder = mock_agent.actor.encoder

    if Adapt.encoder_from_scratch:
        logger.print('Adapt.encoder_from_scratch is set. Randomizing weights of encoder.')
        from iti.dmc_gen.algorithms.modules import weight_init
        encoder.apply(weight_init)
    config = dict(
        encoder=encoder, actor_from_obs=DMCGENDummyActor(mock_agent.actor),
        algorithm=Adapt.agent_stem, action_shape=action_shape,
        feature_dim=Args.projection_dim  # TODO: <-- check if this is reasonable!
    )

    from iti.invr_thru_inf.dynamics import (DynamicsModel,
                                            InverseDynamicsHead, ForwardDynamicsHead, ImplicitDynamicsHead,
                                            invm_loss, fwdm_loss, impm_loss)
    from torch.optim import RMSprop
    if Adapt.implicit_model:
        impm_head = ImplicitDynamicsHead(Args.projection_dim, Adapt.adapt_hidden_dim).to(Args.device)
        dynamics_model = DynamicsModel(
            heads=(impm_head,),
            optims=(RMSprop(impm_head.parameters(), lr=Adapt.impm_pretrain_lr),),
            enc_optims=(RMSprop(encoder.parameters(), lr=Adapt.impm_lr),),
            loss_funcs=(impm_loss, ),
            head_key=('impm_loss', )
        )
    else:
        inv_head = InverseDynamicsHead(Args.projection_dim * 2,
                                    action_shape[0],
                                    Adapt.adapt_hidden_dim).to(Args.device)
        fwd_head = ForwardDynamicsHead(Args.projection_dim,
                                    action_shape[0],
                                    Adapt.adapt_hidden_dim).to(Args.device)
        dynamics_model = DynamicsModel(
            heads=(inv_head, fwd_head),
            optims=(RMSprop(inv_head.parameters(), lr=Adapt.invm_pretrain_lr),
                    RMSprop(fwd_head.parameters(), lr=Adapt.fwdm_pretrain_lr)),
            enc_optims=(RMSprop(encoder.parameters(), lr=Adapt.invm_lr),
                        RMSprop(encoder.parameters(), lr=Adapt.fwdm_lr)),
            loss_funcs=(invm_loss, fwdm_loss),
            head_key=('invm', 'fwdm')
        )

    adapt_agent = AdaptationAgent(
        **config,
        hidden_dim=Adapt.adapt_hidden_dim,
        dynamics_model=dynamics_model,
        device=Args.device
    )
    logger.print('done')

    return adapt_agent


def main(**kwargs):
    from iti.helpers.tticslurm import set_egl_id
    from iti.invr_thru_inf.utils import set_seed_everywhere
    from iti.invr_thru_inf.adapt import prepare_buffers, adapt_offline
    from iti.invr_thru_inf.env_helpers import get_env
    from iti.invr_thru_inf.replay_buffer import Replay
    from iti.invr_thru_inf.config import Adapt
    from iti import RUN

    from .config import Args

    from warnings import simplefilter  # noqa
    simplefilter(action='ignore', category=DeprecationWarning)


    mllogger.start('run', 'start')
    # Update config parameters based on kwargs

    Args._update(kwargs)
    Adapt._update(kwargs)

    set_seed_everywhere(kwargs['seed'])

    assert Adapt.snapshot_prefix

    # TODO: This should work fine as I moved all parameters.pkl to GCP bucket.
    params_path = pjoin(Args.checkpoint_root, Adapt.snapshot_prefix, 'parameters.pkl')
    logger.print('reading from', params_path)
    # Check if the job is completed
    try:
        completion_time = mllogger.read_params('job.completionTime', path=params_path)
        logger.print(f'Pre-training was completed at {completion_time}.')
    except KeyError:
        logger.print(f'training for {Adapt.snapshot_prefix} has not been finished yet.')
        logger.print('job.completionTime was not found.')
        raise RuntimeError

    # Update parameters
    keep_args = ['eval_env_name', 'seed', 'tmp_dir', 'checkpoint_root', 'time_limit', 'device']
    Args._update({key: val for key, val in mllogger.read_params("Args", path=params_path).items() if key not in keep_args})
    logger.print('Args after update', vars(Args))

    params_prefix = pjoin(Args.checkpoint_root, RUN.prefix)
    mllogger.log_params(Args=vars(Args), Adapt=vars(Adapt), path=pjoin(params_prefix, 'parameters.pkl'))


    # buffer_env_name = Args.env_name if not Adapt.targ_buffer_name else Adapt.targ_buffer_name
    # targ_buffer_name = Args.eval_env_name
    orig_buffer_dir, targ_buffer_dir = prepare_buffers(
        Args.checkpoint_root, Args.tmp_dir, Args.env_name, Args.eval_env_name,
        Args.action_repeat, Args.seed, Adapt
    )

    # ===== make or load adaptation-agent =====
    checkpoint_path = pjoin(params_prefix, 'checkpoint.pkl')
    if mllogger.glob(checkpoint_path):

        # Verify that arguments are consistent
        from iti.invr_thru_inf.utils import verify_args
        loaded_args = mllogger.read_params("Args")
        verify_args(vars(Args), loaded_args)

        Progress._update(mllogger.read_params(path=checkpoint_path))
        try:
            # Adaptation has been already completed
            completion_time = mllogger.read_params('job.completionTime', path=pjoin(params_prefix, 'parameters.pkl'))
            logger.print(f'job.completionTime is set:{completion_time}\n',
                         'This job seems to have been completed already.')
            return
        except KeyError:
            assert not Adapt.encoder_from_scratch
            from iti.invr_thru_inf.adapt import load_snapshot
            # Load adaptation agent from s3
            logger.print(f'Resuming from the checkpoint. step: {Progress.step}')
            adapt_agent = load_snapshot(Args.checkpoint_root)

            mllogger.start('episode')
            # mllogger.timer_cache['start'] = mllogger.timer_cache['episode'] - Progress.wall_time

    else:
        from iti.invr_thru_inf.config import Adapt
        adapt_agent = load_adpt_agent(Args, Adapt)

    # NOTE: Below only requires Args and Adapt
    latent_replay = Replay(
        buffer_dir=orig_buffer_dir, buffer_size=Adapt.latent_buffer_size,
        batch_size=Adapt.batch_size, num_workers=4
    )
    distr_replay = Replay(
        buffer_dir=targ_buffer_dir, buffer_size=Adapt.latent_buffer_size,
        batch_size=Adapt.batch_size, num_workers=4
    )
    orig_eval_env = get_env(Args.env_name, Args.frame_stack, Args.action_repeat, Args.seed + 1000,
                            size=Args.image_size)
    targ_eval_env = get_env(Args.eval_env_name, Args.frame_stack, Args.action_repeat, Args.seed,
                            distraction_config=Adapt.distraction_types, size=Args.image_size,
                            intensity=Adapt.distraction_intensity)

    logger.print('orig_eval_env', orig_eval_env)
    logger.print('targ_eval_env', targ_eval_env)

    assert Adapt.adapt_offline
    adapt_offline(Adapt,
                  Args.checkpoint_root, Args.action_repeat, Args.time_limit, Args.batch_size, Args.device,
                  orig_eval_env, targ_eval_env, adapt_agent, latent_replay, distr_replay,
                  progress=Progress)

if __name__ == '__main__':
    # Parse arguments
    import argparse
    from iti.invr_thru_inf.config import Adapt
    from .config import Args
    from iti.helpers.tticslurm import prepare_launch

    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_file", type=str,
                        help="sweep file")
    parser.add_argument("-l", "--line-number",
                        type=int, help="line number of the sweep-file")
    args = parser.parse_args()

    # Obtain kwargs from Sweep
    from params_proto.neo_hyper import Sweep
    sweep = Sweep(Args, Adapt).load(args.sweep_file)
    kwargs = list(sweep)[args.line_number]

    # Set prefix
    job_name = kwargs['job_name']
    from iti import RUN
    RUN.prefix = f'{RUN.project}/{job_name}'

    prepare_launch(Args.job_name)

    # Obtain / generate wandb_runid
    params_path = pjoin(Args.checkpoint_root, RUN.prefix, 'parameters.pkl')
    try:
        wandb_runid = mllogger.read_params("wandb_runid", path=params_path)
    except Exception as e:
        print(e)
        wandb_runid = wandb.util.generate_id()
        mllogger.log_params(wandb_runid=wandb_runid, path=params_path)

    logger.info('params_path', params_path)
    logger.info('If wandb initialization fails with "conflict" error, deleting ^ forces to generate a new wandb id (but all intermediate progress will be lost.)')
    logger.print('wandb_runid:', wandb_runid)

    # Initialize wandb
    wandb.login()  # NOTE: You need to set envvar WANDB_API_KEY
    wandb.init(
        # Set the project where this run will be logged
        project=RUN.project + '-adapt',
        # Track hyperparameters and run metadata
        config=kwargs,
        resume=True,
        id=wandb_runid
    )

    if wandb.run.summary.get('completed', False) or mllogger.glob(pjoin(Args.checkpoint_root, RUN.prefix, 'snapshot.pt')):
        logger.print('The job seems to have been completed!!')
        wandb.finish()
        quit()

    wandb.run.summary['completed'] = wandb.run.summary.get('completed', False)

    # Oftentimes crash logs are not viewable on wandb webapp. This uploads the slurm log.
    wandb.save(f'./slurm-{os.getenv("SLURM_JOB_ID")}.out', policy='end')
    wandb.save(f'./slurm-{os.getenv("SLURM_JOB_ID")}.out.log', policy='end')
    wandb.save(f'./slurm-{os.getenv("SLURM_JOB_ID")}.error.log', policy='end')

    # Controlled exit before slurm kills the process
    try:
        main(**kwargs)
        wandb.run.summary['completed'] = True
        wandb.finish()
    except TimeoutError as e:
        wandb.mark_preempting()
        raise e
