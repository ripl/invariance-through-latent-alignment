#!/usr/bin/env python3
from os.path import join as pJoin
import torch

from iti.invr_thru_inf.adapt import Progress

def load_adpt_agent(Args, Agent, Adapt):
    """
    Load from the pretrained agent specified by snapshot_prefix.
    """
    from ml_logger import logger
    from iti.invr_thru_inf.env_helpers import get_env
    from iti.invr_thru_inf.agent import AdaptationAgent

    dummy_env = get_env(Args.train_env, Args.frame_stack, Args.action_repeat, Args.seed)
    action_shape = dummy_env.action_space.shape

    from .dummy_actors import DrQV2DummyActor
    # snapshot_dir = pJoin(Args.checkpoint_root, Adapt.snapshot_prefix)
    # TEMP: Load DrQv2 agent from s3 rather than gs!
    snapshot_dir = pJoin('s3://ge-data-improbable/drqv2_invariance', Adapt.snapshot_prefix)

    snapshot_path = pJoin(snapshot_dir, 'snapshot.pt')
    logger.print('Loading model from', snapshot_path)
    pret_agent = mllogger.load_torch(path=snapshot_path)
    encoder = torch.nn.Sequential(pret_agent.encoder, pret_agent.actor.trunk)
    config = dict(
        encoder=encoder, actor_from_obs=DrQV2DummyActor(pret_agent),
        algorithm='drqv2', action_shape=action_shape,
        feature_dim=Agent.feature_dim
    )

    from iti.invr_thru_inf.dynamics import (DynamicsModel,
                                        InverseDynamicsHead, ForwardDynamicsHead, ImplicitDynamicsHead,
                                        invm_loss, fwdm_loss, impm_loss)
    from torch.optim import RMSprop
    if not Adapt.implicit_model:
        inv_head = InverseDynamicsHead(Agent.feature_dim * 2,
                                    action_shape[0],
                                    Adapt.adapt_hiden_dim).to(Args.device)
        fwd_head = ForwardDynamicsHead(Agent.feature_dim,
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
    else:
        impm_head = ImplicitDynamicsHead(Agent.feature_dim, Adapt.adapt_hidden_dim).to(Args.device)
        dynamics_model = DynamicsModel(
            heads=(impm_head,),
            optims=(RMSprop(impm_head.parameters(), lr=Adapt.impm_pretrain_lr),),
            enc_optims=(RMSprop(encoder.parameters(), lr=Adapt.impm_lr),),
            loss_funcs=(impm_loss, ),
            head_key=('impm_loss', )
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
    from ml_logger import logger, RUN, ML_Logger
    from iti.invr_thru_inf.utils import set_seed_everywhere
    from iti.invr_thru_inf.adapt import prepare_buffers, adapt_offline
    from iti.invr_thru_inf.env_helpers import get_env
    from iti.invr_thru_inf.replay_buffer import Replay
    from iti.invr_thru_inf.config import Adapt
    from iti.helpers.tticslurm import set_egl_id

    from .config import Args, Agent

    from warnings import simplefilter  # noqa
    simplefilter(action='ignore', category=DeprecationWarning)

    if not torch.cuda.is_available():
        raise RuntimeError('torch.cuda.is_available() is False!!')

    logger.start('run', 'start')

    # Update config parameters based on kwargs
    set_egl_id()
    set_seed_everywhere(kwargs['seed'])

    Args._update(kwargs)
    Agent._update(kwargs)
    Adapt._update(kwargs)

    if Adapt.snapshot_prefix:
        src_logger = ML_Logger(
            prefix=Adapt.snapshot_prefix
        )

        assert src___prefix__ != __prefix__
        logger.print('src___prefix__', src___prefix__)
        logger.print('__prefix__', __prefix__)

        # Check if the job is completed
        try:
            completion_time = src_mllogger.read_params('job.completionTime')
            logger.print(f'Pre-training was completed at {completion_time}.')
        except KeyError:
            logger.print(f'training for {__prefix__} has not been finished yet.')
            logger.print('job.completionTime was not found.')
            if not RUN.debug:
                raise RuntimeError

        # Load from the checkpoint
        assert RUN.debug or src_mllogger.glob('checkpoint.pkl')

        # Update parameters
        keep_args = ['eval_env', 'seed', 'tmp_dir', 'checkpoint_root', 'time_limit', 'device']
        Args._update({key: val for key, val in src_mllogger.read_params("Args").items() if key not in keep_args})
        Agent._update(**src_mllogger.read_params('Agent'))
    mllogger.log_params(Args=vars(Args), Agent=vars(Agent), Adapt=vars(Adapt))

    logger.log_text("""
            keys:
            - Args.train_env
            - Args.seed
            charts:
            - yKey: eval/episode_reward
              xKey: step
            - yKey: clean_eval/episode_reward
              xKey: step
            - yKey: encoder_loss/mean
              xKey: step
            - yKeys: ["discriminator_loss/mean", "discr_adpt_loss/mean", "discr_org_loss/mean"]
              xKey: step
            - yKey: pretrain/invm_loss/mean
              xKey: pretrain/step
            - yKey: pretrain/fwdm_loss/mean
              xKey: pretrain/step
            - yKeys: ["pretrain/invm_distr", "pretrain/invm_clean"]
              xKey: pretrain/step
            - yKeys: ["pretrain/fwdm_distr", "pretrain/fwdm_clean"]
              xKey: pretrain/step
            - yKey: invm_loss/mean
              xKey: step
            - yKey: fwdm_loss/mean
              xKey: step
            - yKeys: ["eval_invm/distr", "eval_invm/clean"]
              xKey: step
            - yKeys: ["eval_fwdm/distr", "eval_fwdm/clean"]
              xKey: step
            """, ".charts.yml", overwrite=True, dedent=True)


    buffer_env_name = Args.train_env if not Adapt.targ_buffer_name else Adapt.targ_buffer_name
    targ_buffer_name = Args.eval_env
    from ml_logger import logger
    logger.print('targe buffer name', targ_buffer_name)
    orig_buffer_dir, targ_buffer_dir = prepare_buffers(
        Args.checkpoint_root, Args.tmp_dir, buffer_env_name, targ_buffer_name,
        Args.action_repeat, Args.seed, Adapt
    )

    # ===== make or load adaptation-agent =====
    if RUN.resume and mllogger.glob('checkpoint.pkl'):

        # Verify that arguments are consistent
        from iti.invr_thru_inf.utils import verify_args
        loaded_args = mllogger.read_params("Args")
        verify_args(vars(Args), loaded_args)
        loaded_args = mllogger.read_params("Agent")
        verify_args(vars(Agent), loaded_args)

        Progress._update(mllogger.read_params(path="checkpoint.pkl"))
        try:
            # Adaptation has been already completed
            completion_time = mllogger.read_params('job.completionTime')
            logger.print(f'job.completionTime is set:{completion_time}\n',
                         'This job seems to have been completed already.')
            return
        except KeyError:
            from iti.invr_thru_inf.adapt import load_snapshot
            # Load adaptation agent from s3
            logger.print(f'Resuming from the checkpoint. step: {Progress.step}')
            adapt_agent = load_snapshot(Args.checkpoint_root)

            logger.start('episode')
            mllogger.timer_cache['start'] = mllogger.timer_cache['episode'] - Progress.wall_time

    else:
        from iti.invr_thru_inf.config import Adapt
        adapt_agent = load_adpt_agent(Args, Agent, Adapt)

    # NOTE: Below only requires Args, Agent and Adapt
    latent_replay = Replay(
        buffer_dir=orig_buffer_dir, buffer_size=Adapt.latent_buffer_size,
        batch_size=Adapt.batch_size, num_workers=4,
        store_states=False,
    )
    distr_replay = Replay(
        buffer_dir=targ_buffer_dir,
        buffer_size=Adapt.latent_buffer_size,
        batch_size=Adapt.batch_size,
        num_workers=4,
        store_states=False
    )
    orig_eval_env = get_env(Args.train_env, Args.frame_stack, Args.action_repeat, Args.seed + 1000)
    logger.print('eval_env', Args.eval_env)
    targ_eval_env = get_env(Args.eval_env, Args.frame_stack, Args.action_repeat, Args.seed,
                            distraction_config=Adapt.distraction_types,
                            intensity=Adapt.distraction_intensity)

    logger.print('orig_eval_env', orig_eval_env)
    logger.print('targ_eval_env', targ_eval_env)

    assert Adapt.adapt_offline
    adapt_offline(Adapt,
                  Args.checkpoint_root, Args.action_repeat, Args.time_limit, Adapt.batch_size, Args.device,
                  orig_eval_env, targ_eval_env, adapt_agent, latent_replay, distr_replay,
                  progress=Progress)

