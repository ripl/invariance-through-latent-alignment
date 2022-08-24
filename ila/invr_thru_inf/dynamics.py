#!/usr/bin/env python3
import torch
from torch import nn
from .modules import weight_init
import wandb


class InverseDynamicsHead(nn.Module):
    """Now this has the same architecture as Actor MLP"""
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.apply(weight_init)

    def forward(self, latent, next_latent):
        mu = self.mlp(torch.cat([latent, next_latent], dim=-1))
        return torch.tanh(mu)


class ForwardDynamicsHead(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()

        self.latent_input = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)
        )

        self.action_input = nn.Sequential(
            nn.Linear(action_dim, 100), nn.LayerNorm(100),
            nn.Linear(100, 100), nn.ReLU(inplace=True)
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 100, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim), nn.LayerNorm(latent_dim),
            nn.Tanh()
        )
        self.apply(weight_init)

    def forward(self, latent, action):
        action = action.to(torch.float32)  # TEMP: force float action
        merged = torch.cat([self.latent_input(latent), self.action_input(action)], dim=-1)
        return self.mlp(merged)


class ImplicitDynamicsHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        self.apply(weight_init)

    def forward(self, latent, next_latent, action):
        return self.mlp(torch.cat([latent, next_latent, action], dim=-1))


class DynamicsModel:
    def __init__(self, heads, optims, enc_optims, loss_funcs=None, head_key=None) -> None:
        self.heads = heads
        self.optims = optims
        self.enc_optims = enc_optims
        self.loss_funcs = [] if loss_funcs is None else loss_funcs
        self.head_key = [] if head_key is None else head_key

        assert len(self.head_key) == len(self.loss_funcs)

    def update_model(self, orig_latent, next_orig_latent, action, wandb_prefix=''):
        """Update dynamics model. Used during dynamics pretraining."""
        from ml_logger import logger

        for optim in self.optims:
            optim.zero_grad(set_to_none=True)

        losses = [loss_func(head, orig_latent, next_orig_latent, action)
                  for head, loss_func in zip(self.heads, self.loss_funcs)]
        loss = sum(losses)
        loss.backward()

        for optim in self.optims:
            optim.step()

        logdict = {
            'dynamics_loss': loss.item(),
            **{tag + '_loss': loss for tag, loss in zip(self.head_key, losses)}
        }

            # wandb.log(
            #     {'/'.join((wandb_prefix, key)) if wandb_prefix else key: val for key, val in logdict.items()}
            # )
            # logger.store_metrics(
            #     dynamics_loss=loss.item(),
            #     **{tag: loss for tag, loss in zip(self.head_key, losses)}
            # )
            # logger.store_metrics(dynamics_loss=loss.item())

            # wandb.log({'dynamics_loss': loss.item()})
        wandb.log({wandb_prefix: logdict} if wandb_prefix else logdict)

    def update_encoder(self, obs, next_obs, action, encode, wandb_prefix=''):
        """forward is a nn.Module that encodes the inputs"""
        from ml_logger import logger

        for enc_optim in self.enc_optims:
            enc_optim.zero_grad(set_to_none=True)

        orig_latent, next_orig_latent, action = encode(obs, next_obs, action)
        losses = [loss_func(head, orig_latent, next_orig_latent, action)
                  for head, loss_func in zip(self.heads, self.loss_funcs)]
        loss = sum(losses)
        loss.backward()

        for enc_optim in self.enc_optims:
            enc_optim.step()

        logdict = {
            'dynamics_loss': loss.item(),
            **{tag + '_loss': loss for tag, loss in zip(self.head_key, losses)}
        }
        wandb.log({wandb_prefix: logdict} if wandb_prefix else logdict)
        # wandb.log(
        #     {'/'.join((wandb_prefix, key)) if wandb_prefix else key: val for key, val in logdict.items()}
        # )
        # if self.head_key:
        #     assert len(self.head_key) == len(self.loss_funcs)
        #     logger.store_metrics(
        #         dynamics_loss=loss.item(),
        #         **{tag: loss for tag, loss in zip(self.head_key, losses)}
        #     )
        # else:
        #     logger.store_metrics(dynamics_loss=loss.item())

    def train(self, training):
        for head in self.heads:
           head.train(training)


def invm_loss(head, latent, next_latent, action):
    from torch.nn import functional as F
    pred_action = head(latent, next_latent)
    return F.mse_loss(pred_action, action)

def fwdm_loss(head, latent, next_latent, action):
    from torch.nn import functional as F
    pred_latent = head(latent, action)
    return F.mse_loss(pred_latent, next_latent)

def impm_loss(head, latent, next_latent, action):
    """Loss for implicit dynamics_model model.

    Negative samples are randomly shuffled triplet of the original input.
    The loss is simple binary cross entropy, assuming that impm_head has sigmoid at the last layer.
    """
    pos_score = head(latent, next_latent, action)

    # TODO: Will gradients backprop properly with shuffled copy??
    random_idx0 = torch.randperm(latent.size(0))
    random_idx1 = torch.randperm(latent.size(0))
    neg_next_latent = next_latent.copy()[random_idx0]
    neg_action = action.copy()[random_idx1]

    return - torch.log(pos_score) - torch.log(1 - neg_score)
