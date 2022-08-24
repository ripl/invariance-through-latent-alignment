import torch
import torch.nn.functional as F
from . import modules as m
from .sac import SAC


class CURL(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.aux_update_freq = args.aux_update_freq

        self.curl_head = m.CURLHead(self.critic.encoder).cuda()

        self.curl_optimizer = torch.optim.Adam(
            self.curl_head.parameters(), lr=args.aux_lr, betas=(args.aux_beta, 0.999)
        )
        self.train()

    def train(self, training=True):
        super().train(training)
        if hasattr(self, 'curl_head'):
            self.curl_head.train(training)

    def update_curl(self, x, x_pos):
        assert x.size(-1) == 84 and x_pos.size(-1) == 84
        from ml_logger import logger

        z_a = self.curl_head.encoder(x)
        with torch.no_grad():
            z_pos = self.critic_target.encoder(x_pos)

        logits = self.curl_head.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().cuda()
        curl_loss = F.cross_entropy(logits, labels)

        self.curl_optimizer.zero_grad()
        curl_loss.backward()
        self.curl_optimizer.step()

        logger.store_metrics({'train/aux_loss': curl_loss.item()})

    def update(self, replay_buffer, step):
        obs, action, reward, next_obs, not_done, pos = replay_buffer.sample_curl()

        self.update_critic(obs, action, reward, next_obs, not_done)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

        if step % self.aux_update_freq == 0:
            self.update_curl(obs, pos)
