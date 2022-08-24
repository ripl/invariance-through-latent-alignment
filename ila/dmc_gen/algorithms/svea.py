import torch
import torch.nn.functional as F
from .. import utils
from .. import augmentations
from .sac import SAC
import wandb


class SVEA(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.svea_alpha = args.svea_alpha
        self.svea_beta = args.svea_beta

    def update_critic(self, obs, action, reward, next_obs, not_done):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        if self.svea_alpha == self.svea_beta:
            obs = utils.cat(obs, augmentations.random_conv(obs.clone()))
            action = utils.cat(action, action)
            target_Q = utils.cat(target_Q, target_Q)

            current_Q1, current_Q2 = self.critic(obs, action)
            critic_loss = (self.svea_alpha + self.svea_beta) * (
                    F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
        else:
            current_Q1, current_Q2 = self.critic(obs, action)
            critic_loss = self.svea_alpha * (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

            obs_aug = augmentations.random_conv(obs.clone())
            current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
            critic_loss += self.svea_beta * (
                    F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

        # logger.store_metrics({'train_critic/loss': critic_loss.item()})
        wandb.log({'critic/loss': critic_loss.item()})

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update(self, replay_buffer, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample_drq()

        self.update_critic(obs, action, reward, next_obs, not_done)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
