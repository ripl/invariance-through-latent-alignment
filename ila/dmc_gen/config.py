import torch.cuda
from params_proto.neo_proto import ParamsProto, Proto, Flag, PrefixProto


class Args(ParamsProto):
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # environment
    seed = None
    env_name = 'dmc:Walker-walk-v1'
    eval_env_name = 'dmc:Walker-walk-v1'

    frame_stack = 3
    action_repeat = 4
    episode_length = 1000

    # agent
    algorithm = 'sac'
    train_steps = 500_000
    discount = 0.99
    init_steps = 1000
    batch_size = 128
    hidden_dim = 1024
    image_size = 84
    image_crop_size = 84

    # actor
    actor_lr = 1e-3
    actor_beta = 0.9
    actor_log_std_min = -10
    actor_log_std_max = 2
    actor_update_freq = 2

    # critic
    critic_lr = 1e-3
    critic_beta = 0.9
    critic_tau = 0.01
    critic_target_update_freq = 2

    # architecture
    num_shared_layers = 11
    num_head_layers = 0
    num_filters = 32
    projection_dim = 100
    encoder_tau = 0.05

    # entropy maximization
    init_temperature = 0.1
    alpha_lr = 1e-4
    alpha_beta = 0.5

    # auxiliary tasks
    aux_lr = 1e-3
    aux_beta = 0.9
    aux_update_freq = 2

    # soda
    soda_batch_size = 256
    soda_tau = 0.005

    # svea
    svea_alpha = 0.5
    svea_beta = 0.5

    # eval: all freq in steps
    eval_freq = 10_000  # default: 10_000
    log_freq = 1_000
    checkpoint_freq = 200_000
    # save_freq = 200_000
    eval_episodes = 30

    # distracting-control envs
    distracting_cs_intensity = 0.

    # gym-fetch
    cam_distance = 1.0
    cam_azimuth = 180
    cam_elevation = -30

    tmp_dir = Proto(env="CUSTOM_TMPDIR")

    # misc
    checkpoint_root = Proto(None, env="SNAPSHOT_ROOT",
                            help="upload snapshots here at the end, used to resuming after preemption. "
                                 "it can be a remote directory (e.g., s3://bucket-name/foo/bar).")

    # example: /share/data/ripl/takuma/dataset/places365_standard
    places_dataset_path = Proto(env="PLACES_DATASET_PATH",
                                help="root directory for places dataset. This is used in SODA augmentation.")

    time_limit = Proto(60 * 60 * 40, env="TIME_LIMIT", dtype=int, help="time limit before halting and upload snapshot"
                       "3.7 * 60 * 60 ~ 3h42min (Copying 20GB+ buffers to NFS may take some time)")

    save_video = Flag(default=True)

    job_name = ''
