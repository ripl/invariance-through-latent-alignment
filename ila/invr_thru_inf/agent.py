#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils
from .modules import RLProjection, weight_init
import wandb


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class Discriminator(nn.Module):
    def __init__(self, repr_dim, feature_dim, hidden_dim):
        super().__init__()
        # NOTE: SODA uses m.SODAMLP. Would that be better??
        self.trunk = RLProjection(repr_dim, feature_dim)

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(weight_init)

    def forward(self, obs):
        h = self.trunk(obs)
        mu = self.mlp(h)
        return mu


class AdaptationAgent:
    def __init__(
            self, algorithm, encoder, actor_from_obs, action_shape, feature_dim, hidden_dim, dynamics_model, device
    ):
        from copy import deepcopy

        self.algorithm = algorithm
        self.encoder = encoder
        self.actor_from_obs = actor_from_obs
        self.device = device
        self.training = False

        self.dynamics_model = dynamics_model

        from .config import Adapt

        # Set up optimizers for encoder
        # self.encoder_opt = torch.optim.RMSprop(self.encoder.parameters(), lr=Adapt.gan_lr)
        self.encoder_opt = torch.optim.RMSprop(self.encoder.parameters(), lr=Adapt.gan_enc_lr)

        self.discriminator = Discriminator(feature_dim, feature_dim, hidden_dim).to(device)
        self.discriminator_opt = torch.optim.RMSprop(self.discriminator.parameters(), lr=Adapt.gan_disc_lr)
        self.discr_weight_clip = Adapt.discr_weight_clip

        self.discriminator_score = nn.Sequential(
            deepcopy(self.discriminator),
            nn.Sigmoid()
        )
        self.discriminator_score_opt = torch.optim.RMSprop(self.discriminator_score.parameters(), lr=Adapt.gan_disc_lr)
        self.augment = Adapt.augment

        if Adapt.augment:
            self.aug = RandomShiftsAug(pad=4)

    def _calculate_gradient_penalty(self, original_latent, adapting_latent):
        # NOTE: Improved training of WGANs
        # Taken from https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py
        alpha = torch.rand(original_latent.shape[0], 1, device=self.device)
        alpha = alpha.expand(original_latent.size())

        # Need to set requires_grad to True to run autograd.grad
        interpolates = (alpha * original_latent + (1 - alpha) * adapting_latent).detach().requires_grad_(True)

        # Calculate gradient penalty
        discr_interpolates = self.discriminator(interpolates)

        gradients = torch.autograd.grad(outputs=discr_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(discr_interpolates.size(), device=self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def update_discriminator(self, replay_iter, adapting_latent, improved_wgan=False):
        """Update discriminator
        ~maximize: log(D(z)) + log(1 - D(G(x)))~
        maximize: D(z) - D(G(x))
        z: nondistracting latent
        x: distracting observation
        (Real) nondistr_label: 1
        (Fake) distr_label: -1

        Steps:
        1. optim = optimizer(discriminator.params())
        2. discriminator.zero_grad()
        3. $z_t^\text{clean}$ -> discriminator -> loss_nondistr
        4. loss_nondistr.backward()
        5. $o_t^\text{distr}$ -> encoder -> $z_t^\text{distr}$ -> discriminator -> loss_distr
        6. loss_distr.backward()
        7. optim.step()
        """
        self.discriminator_opt.zero_grad(set_to_none=True)

        batch = next(replay_iter)
        original_latent = utils.to_torch(batch, self.device)[0]

        if self.augment:
            # Sample one out of four in each instance
            # NOTE: original_latent: (batch_size, 4, 1024)
            batch_size, num_augments, _ = original_latent.shape
            original_latent = original_latent[range(batch_size), torch.randint(0, num_augments - 1, (batch_size, )), :]


        # standard wGAN discriminator loss
        loss_org = - self.discriminator(original_latent).mean()
        loss_adpt = self.discriminator(adapting_latent).mean()
        loss = loss_org + loss_adpt

        if improved_wgan:
            grad_penalty_coef = 1000
            gradient_penalty = self._calculate_gradient_penalty(original_latent, adapting_latent)
            loss += gradient_penalty * grad_penalty_coef
            wandb.log({
                'wgan/grad_penalty':gradient_penalty.item()
            }, commit=False)

        loss.backward()
        self.discriminator_opt.step()

        if not improved_wgan:
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.discr_weight_clip, self.discr_weight_clip)

        wandb.log({
            'discr/orig_loss': loss_org.item(),
            'discr/adpt_loss': loss_adpt.item(),
            'discr/loss': loss.item()
        })

        # Update discriminator_score
        from .config import Adapt
        if Adapt.use_discr_score:
            self.discriminator_score_opt.zero_grad(set_to_none=True)
            org_score = self.discriminator_score(original_latent)
            score_loss_org = F.binary_cross_entropy(org_score, torch.ones_like(org_score))
            adpt_score = self.discriminator_score(adapting_latent)
            score_loss_adpt = F.binary_cross_entropy(adpt_score, torch.zeros_like(adpt_score))
            score_loss = score_loss_org + score_loss_adpt
            score_loss.backward()
            self.discriminator_score_opt.step()

            wandb.log({
                'discr_score/orig_loss': loss_org.item(),
                'discr_score/adpt_loss': loss_adpt.item(),
                'discr_score/loss': loss.item()
            })

    def update_encoder(self, original_obs):
        """Update Generator (Encoder)
        ~minimize log(1 - D(G(x))) <==> maximize log(D(G(x)))~
        maximize D(G(x))
        (Fake) x: distracting observation
        nondistr_label: 1
        distr_label: -1

        1. optim = optimizer(encoder.params())
        2. encoder.zero_grad()
        2. $o_t^\text{distr}$ -> encoder -> $z_t^\text{distr}$ -> discriminator -> loss_distr
        3. loss_distr.backward()
        4. optim.step()
        """

        # augment
        if self.augment:
            original_obs = self.aug(original_obs.float())

        self.encoder_opt.zero_grad(set_to_none=True)

        # Reduce unnecessary computation
        for p in self.discriminator.parameters():
            p.requires_grad = False

        loss = - self.discriminator(self.encoder(original_obs)).mean()  # NOTE: Minimize the loss that is against the favor of discriminator!
        loss.backward()

        self.encoder_opt.step()

        for p in self.discriminator.parameters():
            p.requires_grad = True

        wandb.log({'enc/loss': loss.item()})

    def learn_dynamics(self, orig_latent, next_orig_latent, action):
        """Pretrain dynamics consistency models."""
        if len(orig_latent.shape) == 3:
            # Sample one out of four in each instance
            # NOTE: original_latent: (batch_size, 4, 50)
            batch_size, num_augments, _ = orig_latent.shape
            orig_latent = orig_latent[range(batch_size), torch.randint(0, num_augments - 1, (batch_size, )), :]
            next_orig_latent = next_orig_latent[range(batch_size), torch.randint(0, num_augments - 1, (batch_size, )), :]

        self.dynamics_model.update_model(orig_latent, next_orig_latent, action)

    def act(self, obs):
        obs = torch.as_tensor(obs, device=self.device)
        dist = self.actor_from_obs(obs)
        action = dist.mean

        return action.cpu().numpy()[0]

    def encode(self, obs, to_numpy=False, augment=False):
        obs = torch.as_tensor(obs, device=self.device)

        if len(obs.shape) < 4:
            obs = obs.unsqueeze(0)

        # augment
        if augment:
            obs = self.aug(obs.float())

        with torch.no_grad(), utils.eval_mode(self):
            out = self.encoder(obs)
            if to_numpy:
                out = out.cpu().numpy()
            return out

    def update_encoder_with_dynamics(self, obs, action, next_obs, update_model=False):
        """Update encoder with the gradients from dynamics consistency loss

        - update_model (bool): also update the dynamics model if True
        """
        def _encode(_obs, _next_obs, _action):
            latent = self.encoder(_obs)
            with torch.no_grad():
                next_latent = self.encoder(_next_obs)
            return (latent, next_latent, _action)

        # augment
        if self.augment:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())

        self.dynamics_model.update_encoder(obs, next_obs, action, encode=_encode, wandb_prefix='adapt-enc')

        if update_model:
            self.dynamics_model.update_model(*_encode(obs, next_obs, action), wandb_prefix='adapt-dyn')

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor_from_obs.train(training)
        self.discriminator.train(training)
        self.dynamics_model.train(training)
