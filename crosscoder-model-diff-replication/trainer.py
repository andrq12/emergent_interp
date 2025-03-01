import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import tqdm

from utils import *
from crosscoder import CrossCoder
from buffer import Buffer

class Trainer:
    def __init__(self, cfg, model_A, model_B, all_tokens):
        self.cfg = cfg

        # Models stay on whatever device they were loaded on:
        self.model_A = model_A
        self.model_B = model_B

        # Create CrossCoder, placing it by default on model_A's device (or wherever you choose)
        # If you prefer to specify directly, you can .to("cuda:0") or similar here
        self.crosscoder = CrossCoder(cfg).to("cuda:0")#(next(self.model_A.parameters()).device)

        # We'll grab the crosscoder device for later use:
        self.crosscoder_device = next(self.crosscoder.parameters()).device

        self.buffer = Buffer(cfg, self.model_A, self.model_B, all_tokens)
        self.total_steps = cfg["num_tokens"] // cfg["batch_size"]

        self.optimizer = torch.optim.Adam(
            self.crosscoder.parameters(),
            lr=cfg["lr"],
            betas=(cfg["beta1"], cfg["beta2"]),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self.lr_lambda
        )
        self.step_counter = 0

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=cfg.get("tensorboard_log_dir", "logs"))

    def lr_lambda(self, step):
        """Learning rate schedule"""
        if step < 0.8 * self.total_steps:
            return 1.0
        else:
            return 1.0 - (step - 0.8 * self.total_steps) / (0.2 * self.total_steps)

    def get_l1_coeff(self):
        """Linearly increase from 0 -> l1_coeff over first 5% of steps"""
        if self.step_counter < 0.05 * self.total_steps:
            return self.cfg["l1_coeff"] * self.step_counter / (0.05 * self.total_steps)
        else:
            return self.cfg["l1_coeff"]

    def step(self):
        # 1. Fetch the next batch of activations
        acts = self.buffer.next()
        
        # # 2. Move them all to the CrossCoder's device
        # #    (No need to separate "A" or "B"; CrossCoder just needs them on its own device)
        # for key in acts.keys():
        #     acts[key] = acts[key].to(self.crosscoder_device, non_blocking=True)

        # 3. Compute losses on the CrossCoder device
        losses = self.crosscoder.get_losses(acts)
        loss = losses.l2_loss + self.get_l1_coeff() * losses.l1_loss

        # 4. Backprop & optimization
        loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        # 5. Prepare loss logging
        loss_dict = {
            "loss": loss.item(),
            "l2_loss": losses.l2_loss.item(),
            "l1_loss": losses.l1_loss.item(),
            "l0_loss": losses.l0_loss.item(),
            "l1_coeff": self.get_l1_coeff(),
            "lr": self.scheduler.get_last_lr()[0],
            "explained_variance": losses.explained_variance.mean().item(),
            "explained_variance_A": losses.explained_variance_A.mean().item(),
            "explained_variance_B": losses.explained_variance_B.mean().item(),
        }
        self.step_counter += 1

        return loss_dict

    def log(self, loss_dict):
        """Log each entry in loss_dict to TensorBoard."""
        for key, value in loss_dict.items():
            self.writer.add_scalar(key, value, self.step_counter)

        # Optional: print to console
        print(loss_dict)

    def save(self):
        self.crosscoder.save()

    def train(self):
        self.step_counter = 0
        try:
            for i in tqdm(range(self.total_steps)):
                loss_dict = self.step()
                if i % self.cfg["log_every"] == 0:
                    self.log(loss_dict)
                if (i + 1) % self.cfg["save_every"] == 0:
                    self.save()
        finally:
            self.save()
            self.writer.close()  # Close writer to properly finish logging
