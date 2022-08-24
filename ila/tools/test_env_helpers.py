#!/usr/bin/env python3
"""A simple demo that produces an image from the environment."""
import os
import gym

def save_video(frame_stack, path, fps=20, **imageio_kwargs):
    import tempfile, imageio  # , logging as py_logging
    import shutil
    # py_logging.getLogger("imageio").setLevel(py_logging.WARNING)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    format = 'mp4'
    with tempfile.NamedTemporaryFile(suffix=f'.{format}') as ntp:
        from skimage import img_as_ubyte
        try:
            imageio.mimsave(ntp.name, img_as_ubyte(frame_stack), format=format, fps=fps, **imageio_kwargs)
        except imageio.core.NeedDownloadError:
            imageio.plugins.ffmpeg.download()
            imageio.mimsave(ntp.name, img_as_ubyte(frame_stack), format=format, fps=fps, **imageio_kwargs)
        ntp.seek(0)
        shutil.copy(ntp.name, path)


def get_video(env, ep_len=50, size=256):
    done = False
    obs = env.reset()
    obs_frames = [obs]
    rend_frames = [env.unwrapped.env.physics.render(height=size, width=size, camera_id=0)]
    for _ in range(ep_len):
        act = env.action_space.sample()
        obs, rew, done, info = env.step(act)
        obs_frames.append(obs)
        rend_frames.append(env.unwrapped.env.physics.render(height=size, width=size, camera_id=0))
        if done: break

    return obs_frames, rend_frames

def use_env_helpers(domain, task, distraction, outdir):
    from iti.invr_thru_inf.env_helpers import get_env

    name = f'{domain.capitalize()}-{task}'
    envname = f'distracting_control:{name}-intensity-v1'
    framestack = 3
    action_repeat = 3
    seed = 0
    distraction_config = (distraction, )

    env = get_env(envname, framestack, action_repeat,
                seed, distraction_config, intensity=1.0)

    obs_frames, rend_frames = get_video(env, ep_len=60)
    # save_video(obs_frames, f'{outdir}/distraction_demo/helper/{domain}/{distraction}-obs.mp4')
    save_video(rend_frames, f'{outdir}/distraction_demo/helper/{domain}/{distraction}-rend.mp4')



def main(domain, task, distraction, outdir):
    distr_type = ('background', 'color', 'camera') if distraction == 'all' else (distraction, )
    env = gym.make(
        f'distracting_control:{domain.capitalize()}-{task}-intensity-v1',
        from_pixels=True, channels_first=False, dynamic=True, fix_distraction=True,
        distraction_types=distr_type, disable_zoom=True, distraction_seed=0,
        background_data_path=os.environ.get("DC_BG_PATH", None),
        intensity=1.0, background_kwargs={'video_alpha': 1.0, 'ground_plane_alpha': 0.5}  # Necessary if you use intensity
    )
    obs_frames, rend_frames = get_video(env, ep_len=60)
    save_video(obs_frames, f'{outdir}/distraction_demo/direct2/{domain}/{distraction}-obs.mp4')
    save_video(rend_frames, f'{outdir}/distraction_demo/direct2/{domain}/{distraction}-rend.mp4')


if __name__ == '__main__':
    import os
    from os.path import join as pjoin
    outdir = pjoin(os.getenv('LMD_OUTPUT_DIR'))
    os.makedirs(outdir, exist_ok=True)
    domain = 'cheetah'
    task = 'run'
    distraction = 'video-background'
    # main(domain, task, distraction, outdir)

    use_env_helpers(domain, task, distraction, outdir)
