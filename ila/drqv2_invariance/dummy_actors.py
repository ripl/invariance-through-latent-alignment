#!/usr/bin/env python3
import torch

class DummyDistribution:
    def __init__(self, mu, pi):
        self._mu = mu
        self._pi = pi

    @property
    def mean(self):
        return self._mu

class DMCGENDummyActor(torch.nn.Module):
    """Dummy actor module that takes image observation as input"""
    def __init__(self, actor):
        super().__init__()
        self._actor = actor

    def forward(self, obs):
        mu, pi, log_pi, log_std = self._actor(obs.unsqueeze(0), detach=True)
        return DummyDistribution(mu, pi)

class DrQV2DummyActor(torch.nn.Module):
    """Dummy actor module that takes image observation as input"""
    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def forward(self, obs):
        latent = self._agent.encoder(obs.unsqueeze(0))
        return self._agent.actor(latent, std=0)  # NOTE: we don't need stddev during evaluation
