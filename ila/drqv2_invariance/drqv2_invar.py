# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils
from . import modules as m


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


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(m.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        # NOTE: SODA uses m.SODAMLP. Would that be better??
        self.trunk = m.RLProjection(repr_dim, feature_dim)

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(m.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        # NOTE: SODA uses m.SODAMLP. Would that be better??
        self.trunk = m.RLProjection(repr_dim, feature_dim)

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(m.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2

class Discriminator(nn.Module):
    def __init__(self, repr_dim, feature_dim, hidden_dim):
        super().__init__()
        # NOTE: SODA uses m.SODAMLP. Would that be better??
        self.trunk = m.RLProjection(repr_dim, feature_dim)

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(m.weight_init)

    def forward(self, obs):
        h = self.trunk(obs)
        mu = self.mlp(h)
        return mu


class SelfSupervisedHead(nn.Module):
    def __init__(self, repr_dim, output_size, feature_dim, hidden_dim):
        super().__init__()
        # NOTE: SODA uses m.SODAMLP. Would that be better??
        self.trunk = m.RLProjection(repr_dim, feature_dim)

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_size)
        )
        self.apply(m.weight_init)

    def forward(self, obs):
        h = self.trunk(obs)
        mu = self.mlp(h)
        return mu


class InverseDynamicsHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        self.apply(m.weight_init)

    def forward(self, latent, next_latent):
        return self.mlp(torch.cat([latent, next_latent], dim=-1))


class DrQV2Agent:
    def __init__(self, algorithm, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 stddev_schedule, stddev_clip,
                 inv_coef, rew_coef, num_shared_layers, num_filters, num_head_layers
                 ):
        from . import modules as m
        from .config import Agent
        assert algorithm == 'drqv2'

        self.device = device
        self.critic_target_tau = critic_target_tau
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        shared_cnn = m.SharedCNN(obs_shape, num_shared_layers, num_filters).to(device)
        head_cnn = m.HeadCNN(shared_cnn.out_shape, num_head_layers, num_filters).to(device)
        self.encoder = m.Encoder(shared_cnn, head_cnn)

        self.actor = Actor(self.encoder.out_dim, action_shape, feature_dim,
                           hidden_dim).to(device)
        self.critic = Critic(self.encoder.out_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.out_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # self-supervised tasks
        self.inv_coef = inv_coef
        self.rew_coef = rew_coef
        if inv_coef:
            self.inv_dynamics_head = SelfSupervisedHead(
                self.encoder.out_dim * 2, action_shape[0], feature_dim, hidden_dim
            ).to(device)
            self.inv_dynamics_head.train()
            self.inv_dynamics_opt = torch.optim.Adam(self.inv_dynamics_head.parameters(), lr=lr)
        if rew_coef:
            self.reward_pred_head = SelfSupervisedHead(
                self.encoder.out_dim * 2, 1, feature_dim, hidden_dim
            ).to(device)
            self.reward_pred_head.train()
            self.reward_pred_opt = torch.optim.Adam(self.reward_pred_head.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    # todo: remove step from this method.
    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            # todo: remove this logic, use random flag instead. This is bad code.
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)

        from ml_logger import logger
        logger.store_metrics(stddev=stddev)
        return action.cpu().numpy()[0]

    def update_opt(self, loss, optimizers):
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()


    def update_critic(self, obs, action, reward, discount, next_obs, step, ss_loss=0):

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # optimize encoder, critic and optionally self-supervision heads
        loss = critic_loss + ss_loss

        optimizers = [self.encoder_opt, self.critic_opt]
        if self.rew_coef:
            optimizers.append(self.reward_pred_opt)
        if self.inv_coef:
            optimizers.append(self.inv_dynamics_opt)

        self.update_opt(loss, optimizers)

        from ml_logger import logger
        logger.store_metrics(
            critic_target_q=target_Q.mean().item(),
            critic_q1=Q1.mean().item(),
            critic_q2=Q2.mean().item(),
            critic_loss=critic_loss.item(),
        )

    def update_actor(self, obs, step):

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.update_opt(actor_loss, [self.actor_opt])

        from ml_logger import logger
        logger.store_metrics(
            actor_loss=actor_loss.item(),
            actor_logprob=log_prob.mean().item(),
            actor_ent=dist.entropy().sum(dim=-1).mean().item()
        )

    def update(self, replay_iter, step):

        batch = next(replay_iter)
        # obs: [256, 9, 84, 84], action: [256, 1], reward: [256, 1], discount: [256, 1]
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())

        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        # self-supervised tasks
        ss_loss = self.self_supervision_loss(obs, action, reward, next_obs)

        # update critic
        self.update_critic(obs, action, reward, discount, next_obs, step, ss_loss)

        # update actor
        self.update_actor(obs.detach(), step)

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        from ml_logger import logger
        logger.store_metrics(batch_reward=reward.mean().item())

    def self_supervision_loss(self, obs, action, reward, next_obs, coef=1.0):
        if not self.inv_coef:
            return 0

        pred_action = self.inv_dynamics_head(torch.cat([obs, next_obs], dim=1))
        inv_loss = self.inv_coef * (coef * (action - pred_action) ** 2).mean()
        return inv_loss

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


class AdaptationAgent:
    def __init__(
            self, algorithm, encoder, actor_from_obs, invm_head, action_shape, feature_dim, hidden_dim, device
    ):
        from copy import deepcopy

        self.algorithm = algorithm
        self.encoder = encoder
        self.actor_from_obs = actor_from_obs
        self.device = device
        self.training = False

        if invm_head:
            self.invm_head = invm_head.to(device)
        else:
            self.invm_head = InverseDynamicsHead(feature_dim * 2, action_shape[0], hidden_dim).to(device)

        from .config import Adapt
        self.invm_opt = torch.optim.RMSprop(self.invm_head.parameters(), lr=Adapt.invm_pretrain_lr)

        # Set up optimizers for encoder
        self.encoder_opt = torch.optim.RMSprop(self.encoder.parameters(), lr=Adapt.gan_lr)
        self.invm_encoder_opt = torch.optim.RMSprop(self.encoder.parameters(), lr=Adapt.invm_lr)

        self.discriminator = Discriminator(feature_dim, feature_dim, hidden_dim).to(device)
        self.discriminator_opt = torch.optim.RMSprop(self.discriminator.parameters(), lr=Adapt.gan_lr)
        self.discr_weight_clip = Adapt.discr_weight_clip

        self.discriminator_score = nn.Sequential(
            deepcopy(self.discriminator),
            nn.Sigmoid()
        )
        self.discriminator_score_opt = torch.optim.RMSprop(self.discriminator_score.parameters(), lr=Adapt.gan_lr)
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
        from ml_logger import logger

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

        loss.backward()
        self.discriminator_opt.step()

        if not improved_wgan:
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.discr_weight_clip, self.discr_weight_clip)

        logger.store_metrics(
            discr_org_loss=loss_org.item(),
            discr_adpt_loss=loss_adpt.item(),
            discriminator_loss=loss.item(),
            grad_penalty=gradient_penalty.item() if improved_wgan else 0.0
        )

        # Update discriminator_score
        self.discriminator_score_opt.zero_grad(set_to_none=True)
        org_score = self.discriminator_score(original_latent)
        score_loss_org = F.binary_cross_entropy(org_score, torch.ones_like(org_score))
        adpt_score = self.discriminator_score(adapting_latent)
        score_loss_adpt = F.binary_cross_entropy(adpt_score, torch.zeros_like(adpt_score))
        score_loss = score_loss_org + score_loss_adpt
        score_loss.backward()
        self.discriminator_score_opt.step()

        logger.store_metrics(
            discr_org_score_loss=score_loss_org.item(),
            discr_adpt_score_loss=score_loss_adpt.item(),
            discriminator_score_loss=score_loss.item()
        )

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
        from ml_logger import logger

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
        logger.store_metrics(encoder_loss=loss.item())

        for p in self.discriminator.parameters():
            p.requires_grad = True

    def get_invm_loss(self, latent, next_latent, action):
        pred_action = self.invm_head(latent, next_latent)
        return F.mse_loss(pred_action, action)

    def update_invm(self, orig_latent, next_orig_latent, action):
        from ml_logger import logger

        # Sample one out of four in each instance
        # NOTE: original_latent: (batch_size, 4, 50)
        batch_size, num_augments, _ = orig_latent.shape
        orig_latent = orig_latent[range(batch_size), torch.randint(0, num_augments - 1, (batch_size, )), :]
        next_orig_latent = next_orig_latent[range(batch_size), torch.randint(0, num_augments - 1, (batch_size, )), :]

        self.invm_opt.zero_grad(set_to_none=True)

        invm_loss = self.get_invm_loss(orig_latent, next_orig_latent, action)
        invm_loss.backward()

        self.invm_opt.step()
        logger.store_metrics(
            invm_loss=invm_loss.item()
        )

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        dist = self.actor_from_obs(obs)
        assert eval_mode
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

    def update_encoder_with_invm(self, obs, action, next_obs):
        from ml_logger import logger

        # augment
        if self.augment:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())

        self.invm_encoder_opt.zero_grad(set_to_none=True)

        latent = self.encoder(obs)
        with torch.no_grad():
            next_latent = self.encoder(next_obs)

        invm_loss = self.get_invm_loss(latent, next_latent, action)
        invm_loss.backward()
        self.invm_encoder_opt.step()
        logger.store_metrics(
            invm_loss=invm_loss.item()
        )

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor_from_obs.train(training)
        self.invm_head.train(training)
        self.discriminator.train(training)
