# Codebase for Invariance Through Latent Alignment

<!-- Source code will be available shortly. -->
<!-- PyTorch implementation of Invariance Through Inference (ITI) -->

**Invariance Through Latent Alignment**

[Takuma Yoneda\*](https://takuma.yoneda.xyz/), [Ge Yang\*](https://www.episodeyang.com/), [Matthew Walter](https://home.ttic.edu/~mwalter/), [Bradly Stadie](https://bstadie.github.io/)

[[Paper]](https://arxiv.org/abs/2112.08526) [[Website]](https://invariance-through-latent-alignment.github.io/)

![method](media/method.mp4)

## Dependency
This codebase requires `ml-logger` and `params_proto`. Please look at `ila/docker/Dockerfile` or `Pipfile` for more dependencies.

## How to use
1. Set Args.checkpoint_root to your local path. This is done by setting `$SNAPSHOT_ROOT` environment variable
   1. Make sure the path looks like `file:///root/subdirectory/subsubdirectory`
2. Download pretrained agents
   1. Coming soon
   2. You can also train agents by yourself and store their weights
3. Generate and save trajectories on source (i.e., non-distracted) and target (i.e., distracted) environments
4. Run adapt.py to perform adaptation

If you find our work useful in your research, please consider citing the paper as follows:

```
@misc{yoneda2021invariance,
      title={Invariance Through Latent Alignment}, 
      author={Takuma Yoneda and Ge Yang and Matthew R. Walter and Bradly Stadie},
      year={2021},
      eprint={2112.08526},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
