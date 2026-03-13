import torch
from pathlib import Path
from collections import defaultdict
from torchvision.utils import make_grid
from torch import amp
import torch.nn.functional as F
from tqdm import tqdm

class VQGANTrainer:

    def __init__(
        self,
        model,
        discriminator,
        loss_fn,
        optimizer_g,
        optimizer_d,
        scheduler_g,
        scheduler_d,
        train_loader,
        val_loader,
        device,
        run_dir: Path,
        config,
        writer_train=None,
        writer_val=None,
        is_main=True,  # Only rank 0 logs
        start_epoch=0,
        best_loss=float("inf"),
    ):

        self.model = model
        self.discriminator = discriminator
        self.loss_fn = loss_fn

        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.run_dir = run_dir
        self.writer_train = writer_train
        self.writer_val = writer_val
        self.is_main = is_main

        self.n_epochs = config.training.n_epochs
        self.eval_freq = config.training.eval_freq

        self.max_adv_weight = config.losses.adv_weight
        self.adv_warmup_epochs = config.losses.adv_warmup

        self.scaler_g = amp.GradScaler("cuda")
        self.scaler_d = amp.GradScaler("cuda")

        self.start_epoch = start_epoch
        self.best_loss = best_loss

    # ---------------------------
    # TRAINING LOOP
    # ---------------------------
    def train(self):
        for epoch in range(self.start_epoch, self.n_epochs):
            adv_weight = self._compute_adv_weight(epoch)

            # Set epoch for distributed sampler
            if hasattr(self.train_loader, "sampler") and isinstance(self.train_loader.sampler, torch.utils.data.DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)

            self._train_epoch(epoch, adv_weight)

            if self.scheduler_g:
                self.scheduler_g.step()
            if self.scheduler_d:
                self.scheduler_d.step()

            if (epoch + 1) % self.eval_freq == 0 and self.is_main:
                val_loss = self._validate(epoch, adv_weight)
                self._save_best_checkpoint(epoch, val_loss)
                self._save_periodic_checkpoint(epoch)

        if self.is_main:
            self._save_final_model()

    # ---------------------------
    # TRAINING EPOCH
    # ---------------------------
    def _train_epoch(self, epoch, adv_weight):
        self.model.train()
        self.discriminator.train()

        epoch_perplexity = 0.0
        epoch_used_codes = 0.0
        num_batches = 0
        quantizer = self.model.module.quantizer if hasattr(self.model, "module") else self.model.quantizer

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=not self.is_main)

        for step, batch in enumerate(pbar):
            images = batch["image"]
            if hasattr(images, "as_tensor"):
                images = images.as_tensor()
            images = images.to(self.device)

            # --- Generator step ---
            self.optimizer_g.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda"):
                recon, q_loss, indices = self.model(images)
                perplexity = quantizer.perplexity.item()
                used_codes = indices.unique().numel()
                epoch_perplexity += perplexity
                epoch_used_codes += used_codes
                num_batches += 1

                g_loss, g_logs = self.loss_fn.generator_loss(
                    discriminator=self.discriminator,
                    adv_weight=adv_weight,
                    images=images,
                    reconstructions=recon,
                    quantization_loss=q_loss,
                )

            self.scaler_g.scale(g_loss).backward()
            self.scaler_g.unscale_(self.optimizer_g)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()

            # --- Discriminator step ---
            d_logs = {}
            if adv_weight > 0:
                self.optimizer_d.zero_grad(set_to_none=True)
                with amp.autocast(device_type="cuda"):
                    d_loss, d_logs = self.loss_fn.discriminator_loss(
                        discriminator=self.discriminator,
                        adv_weight=adv_weight,
                        images=images,
                        reconstructions=recon.detach(),
                    )
                self.scaler_d.scale(d_loss).backward()
                self.scaler_d.unscale_(self.optimizer_d)
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()

            # --- Logging (only main rank) ---
            if self.is_main:
                logs = {**g_logs, **d_logs}
                logs["total_g_loss"] = g_loss.item()
                global_step = epoch * len(self.train_loader) + step
                for k, v in logs.items():
                    self.writer_train.add_scalar(k, v, global_step)
                pbar.set_postfix({"g_loss": f"{g_loss.item():.4f}"})

        # Codebook stats
        avg_perplexity = epoch_perplexity / num_batches
        avg_used_codes = epoch_used_codes / num_batches
        usage_ratio = avg_used_codes / quantizer.quantizer.num_embeddings
        if self.is_main:
            self.writer_train.add_scalar("codebook/perplexity", avg_perplexity, epoch)
            self.writer_train.add_scalar("codebook/used_codes", avg_used_codes, epoch)
            self.writer_train.add_scalar("codebook/usage_ratio", usage_ratio, epoch)

    # ---------------------------
    # VALIDATION
    # ---------------------------
    @torch.no_grad()
    def _validate(self, epoch, adv_weight):
        torch.cuda.empty_cache()
        self.model.eval()
        self.discriminator.eval()

        epoch_perplexity = 0.0
        epoch_used_codes = 0.0
        num_batches = 0
        quantizer = self.model.module.quantizer if hasattr(self.model, "module") else self.model.quantizer

        total = defaultdict(float)
        sample_images, sample_recons = None, None

        for batch_idx, batch in enumerate(self.val_loader):
            images = batch["image"]
            if hasattr(images, "as_tensor"):
                images = images.as_tensor()
            images = images.to(self.device)
            with amp.autocast(device_type="cuda"):
                recon, q_loss, indices = self.model(images)

            perplexity = quantizer.perplexity.item()
            used_codes = indices.unique().numel()
            epoch_perplexity += perplexity
            epoch_used_codes += used_codes
            num_batches += 1

            # Compute only L1 + quantization
            l1 = F.l1_loss(recon, images)
            total["l1_loss"] += l1.item() * images.size(0)
            total["quantization_loss"] += q_loss.mean().item() * images.size(0)

            # Keep first batch for reconstructions
            if batch_idx == 0 and self.is_main:
                sample_images = images[:4].detach().cpu()
                sample_recons = recon[:4].detach().cpu()

            # Free memory
            del recon, q_loss, indices, images

        # Average losses
        for k in total:
            if self.is_main:
                total[k] /= len(self.val_loader.dataset)
                self.writer_val.add_scalar(k, total[k], epoch)

        avg_perplexity = epoch_perplexity / num_batches
        avg_used_codes = epoch_used_codes / num_batches
        usage_ratio = avg_used_codes / quantizer.quantizer.num_embeddings

        if self.is_main:
            self.writer_val.add_scalar("codebook/perplexity", avg_perplexity, epoch)
            self.writer_val.add_scalar("codebook/used_codes", avg_used_codes, epoch)
            self.writer_val.add_scalar("codebook/usage_ratio", usage_ratio, epoch)
            if sample_images is not None and sample_recons is not None:
                self._log_reconstructions(sample_images, sample_recons, epoch)

        val_metric = total["l1_loss"] if self.is_main else None
        if self.is_main:
            print(f"Validation L1: {val_metric:.6f}")
        return val_metric

    # ---------------------------
    # UTILITIES
    # ---------------------------
    def _compute_adv_weight(self, epoch):
        if epoch < self.adv_warmup_epochs:
            return self.max_adv_weight * (epoch / self.adv_warmup_epochs)
        return self.max_adv_weight

    def _save_best_checkpoint(self, epoch, val_loss):
        if val_loss is not None and val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "best_loss": self.best_loss,
                    "model": self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
                    "discriminator": self.discriminator.module.state_dict() if hasattr(self.discriminator, "module") else self.discriminator.state_dict(),
                    "optimizer_g": self.optimizer_g.state_dict(),
                    "optimizer_d": self.optimizer_d.state_dict(),
                },
                self.run_dir / "best_model.pth",
            )

    def _save_periodic_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "best_loss": self.best_loss,
            "model": self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
            "discriminator": self.discriminator.module.state_dict() if hasattr(self.discriminator, "module") else self.discriminator.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
        }

        torch.save(checkpoint, self.run_dir / f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, self.run_dir / "last_checkpoint.pth")
    def _save_final_model(self):
        torch.save(
            self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
            self.run_dir / "final_model.pth",
        )

    def _log_reconstructions(self, images, recons, epoch, n_images=4):
        # Take central slice along depth
        center = images.shape[4] // 2  # last dimension is depth
        images_slice = images[:n_images, :, :, :, center]  # shape [B, C, H, W]
        recons_slice = recons[:n_images, :, :, :, center]
        
        images_grid = make_grid(images_slice, normalize=True, scale_each=True)
        recons_grid = make_grid(recons_slice, normalize=True, scale_each=True)
        
        self.writer_val.add_image("images/ground_truth", images_grid, epoch)
        self.writer_val.add_image("images/reconstruction", recons_grid, epoch)