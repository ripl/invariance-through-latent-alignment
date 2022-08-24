import torch.cuda
from params_proto.neo_proto import ParamsProto, PrefixProto, Proto


class Args(ParamsProto):
    # task settings
    train_env = 'dmc:Cheetah-run-v1'
    eval_env = 'distracting_control:Cheetah-run-easy-v1'
    # env_name = 'dmc:Cheetah-run-v1'  # non-distracting env
    # todo: use dmc:Quadruped-walk-v1 and gym-dmc module.
    frame_stack = 3
    action_repeat = 2
    discount = 0.99
    # train settings
    train_frames = 1_000_000
    num_seed_frames = 4000
    # eval
    eval_every_frames = 10000
    num_eval_episodes = 10
    # replay buffer
    replay_buffer_size = 1000000
    replay_buffer_num_workers = 4
    nstep = 3
    batch_size = 256
    # misc
    seed = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    update_freq = 2
    save_video = True
    log_freq = 100

    # NOTE: If NFS directory is not used, set nfs_dir and local_dir to the same path.
    # Then file transfer between NFS and local filesystem will not be performed.
    # tmp_dir = Proto(None, env="TMPDIR",
    #                 help="directory for tempralily saving replay buffer and snapshot."
    #                      "all the files will be moved under checkpoint_root after the training finishes")
    tmp_dir = Proto(None, env="CUSTOM_TMPDIR",
                    help="directory for tempralily saving replay buffer and snapshot."
                         "all the files will be moved under checkpoint_root after the training finishes")
    checkpoint_root = Proto(None, env="SNAPSHOT_ROOT",
                            help="upload snapshots here at the end, used to resuming after preemption. "
                                 "it can be a remote directory (e.g., s3://bucket-name/foo/bar).")
    time_limit = Proto(None, env="TIME_LIMIT", dtype=int, help="time limit before halting and upload snapshot"
                       "3.7 * 60 * 60 ~ 3h42min (Copying 20GB+ buffers to NFS may take some time)")
    job_name = ''


# todo: merge into Args, pass in explicitly into agent.
class Agent(PrefixProto):
    # NOTE: [pre-training] locations of buffer, agent
    # - buffer (remote): replay_checkpoint = os.path.join(Args.checkpoint_root, __prefix__, 'replay.tar')  (train.py)
    # - buffer (local):  replay_cache_dir = Path(Args.tmp_dir) / __prefix__ / 'replay'  (train.py)
    # - agent (remote):  model_snapshot_path = os.path.join(Args.checkpoint_root ,__prefix__, 'snapshot.pt')  (train.py)

    algorithm = 'drqv2'
    lr = 1e-4
    critic_target_tau = 0.01
    num_expl_steps = 2000
    hidden_dim = 1024
    feature_dim = 100

    num_shared_layers = 11
    num_filters = 32
    num_head_layers = 0

    # TODO: check if this is reasonable.
    # In the paper, authors use different scheduling for different envs.
    stddev_schedule = 'linear(1.0,0.1,500000)'

    stddev_clip = 0.3
    inv_coef = 0.1
    rew_coef = 0.0
