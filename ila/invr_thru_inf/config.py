from params_proto.neo_proto import Proto, PrefixProto

class CollectData(PrefixProto):
    buffer_type = None  # original or target


class Adapt(PrefixProto):
    # NOTE: [adaptation] locations of prefilled-buffer, pretrained-agent, adaptation-agent
    # - prefilled-buffer (remote): pJoin(Args.checkpoint_root, get_buffer_prefix(), 'buffer.tar.gz')  (collect_offline_data.py)
    # - prefilled-buffer (local):  Path(Args.tmp_dir) / 'data-collection' / get_buffer_prefix()  (collect_offline_data.py)
    # - pretrained-agent (remote): pJoin(Args.checkpoint_root, Adapt.snapshot_prefix, 'snapshot.pt')  (upload_checkpoints.py)
    # - adaptation-agent (remote): mllogger.save_torch(agent, path=pJoin(Args.checkpoint_root, __prefix__, 'snapshot.py')) (adapt.py)

    job_name = ''
    distraction_types = ['background']
    distraction_intensity = None

    # example: 'model-free/model-free/takuma/2021/08-30/drqv2_invariance/baseline_slim_projection_envs/train/12.45.09/reacher-easy/i10/r0/100'
    snapshot_prefix = Proto(None, help='a prefix of ML-logger for the pre-trained agent')
    adapt_save_every = 1000
    agent_stem = 'dmc_gen'

    gan_lr = 1e-6  # 1e-6 seems to work the best (at least on Reacher-easy).

    # TEMP:
    gan_enc_lr = 1e-5
    gan_disc_lr = 1e-6

    implicit_model = False
    invm_lr = 1e-6
    invm_pretrain_lr = 1e-3
    fwdm_lr = 1e-6
    fwdm_pretrain_lr = 1e-3

    impm_pretrain_lr = 1e-3
    impm_lr = 1e-6

    latent_buffer_size = 1_000_000
    adapt_log_freq = 50

    adapt_eval_every_steps = 1_000
    num_eval_episodes = 30

    num_adapt_steps = 10_000
    num_discr_updates = 5
    discr_weight_clip = 0.01
    adapt_hidden_dim = 600

    batch_size = 256

    policy_on_clean_buffer = 'random'
    policy_on_distr_buffer = 'random'


    augment = True
    download_buffer = True  # Download pre-filled buffer from S3 bucket

    # Online adaptation related
    num_adapt_seed_frames = -1  # 4000
    adapt_offline = True
    bootstrap_from_offline_buffer = False
    adapt_update_freq = 4
    random_policy = False

    # Not used recently
    improved_wgan = False
    expl_reward = 'discr'
    inv_focal_weight = False

    num_invm_steps = 2000
    invm_log_freq = 100
    invm_eval_freq = 400

    invm_kickin_step = 0

    targ_buffer_name = Proto(None, help="Specify target buffer env name. If None, Args.eval_env_name (or Args.eval_env for drqv2) is used.")
    ur5_traj_dir = Proto(None, env="UR5_FAKE_EPISODES", help="The directory that contains UR5 images collected with jig-zag motion.")
    adapted_snapshot_prefix = Proto(None, help="Use it when you evaluate the adapted policy.")

    encoder_from_scratch = False
    use_discr_score = True

    corrupt_actions = False
    fix_encoder_during_dynpret = True
    adapt_dyn_model = False
