import math
import shutil
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Callable, Optional
from abc import ABC
from tqdm import tqdm

import torch
import torch.distributed as dist
from accelerate import Accelerator, DeepSpeedPlugin, DataLoaderConfiguration
from torch.optim import AdamW
from dataclasses import dataclass, field
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule
from utils.utils import check_bloat16_supported
from vlaos.datasets.data_utils import PaddedCollatorForActionPrediction
from omegaconf import OmegaConf

from vlaos.overwatch import initialize_overwatch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from vlaos.training.metrics import VLAMetrics

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# default deepspeed configs
@dataclass
class DefaultDeepSpeedConfig:
    bf16: dict = field(default_factory=lambda: {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    })
    fp16: dict = field(default_factory=lambda: {
        "enabled": False,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    })
    zero_optimization: dict = field(default_factory=lambda: {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 500000000,
        "contiguous_gradients": True
    })
    train_batch_size: str = "auto"
    train_micro_batch_size_per_gpu: str = "auto"
    wall_clock_breakdown: bool = False

class AcceleratorStrategy(ABC):
    def __init__(
        self,
        vla,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        trainig_stage: str,
        enable_gradient_checkpointing: bool = False,
        enable_mixed_precision_training: bool = True,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.vla = vla
        self.trainig_stage = trainig_stage

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
    ) -> None:
        # save vlm and action_head separately
        # we always save all modules
        if self.accelerator.is_main_process:
            if self.vla.cfg.paradigm == "hierarchical":
                model_state_dicts = {mkey: OrderedDict() for mkey in ["action_head"]}   # only save action head
            else:
                model_state_dicts = {mkey: OrderedDict() for mkey in ["vlm", "action_head"]}

                vlm_unwrapped = self.accelerator.unwrap_model(self.vla.vlm)
                vlm_state_dict = vlm_unwrapped.state_dict()
                for key, param in vlm_state_dict.items():
                    model_state_dicts["vlm"][key] = param

                if "language_planning" in self.vla.cfg.planning_heads:
                    model_state_dicts["language_planning_head"] = OrderedDict()
                    language_planning_head_unwrapped = self.accelerator.unwrap_model(self.vla.language_planning_head)
                    language_planning_head_state_dict = language_planning_head_unwrapped.state_dict()
                    for key, param in language_planning_head_state_dict.items():
                        model_state_dicts["language_planning_head"][key] = param
                        
                if "visual_planning" in self.vla.cfg.planning_heads:
                    model_state_dicts["visual_planning_head"] = OrderedDict()
                    visual_planning_head_unwrapped = self.accelerator.unwrap_model(self.vla.visual_planning_head)
                    visual_planning_head_state_dict = visual_planning_head_unwrapped.state_dict()
                    for key, param in visual_planning_head_state_dict.items():
                        model_state_dicts["visual_planning_head"][key] = param
                        
                if "image_foresight_planning" in self.vla.cfg.planning_heads:
                    model_state_dicts["image_foresight_head"] = OrderedDict()
                    image_foresight_planning_head_unwrapped = self.accelerator.unwrap_model(self.vla.image_foresight_head)
                    image_foresight_planning_head_state_dict = image_foresight_planning_head_unwrapped.state_dict()
                    for key, param in image_foresight_planning_head_state_dict.items():
                        model_state_dicts["image_foresight_head"][key] = param
        
            action_head_unwrapped = self.accelerator.unwrap_model(self.vla.action_head)
            action_head_state_dict = action_head_unwrapped.state_dict()
            for key, param in action_head_state_dict.items():
                model_state_dicts["action_head"][key] = param
                
            checkpoint_dir = run_dir / "checkpoints"
            if train_loss is None:
                checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
            else:
                checkpoint_path = (
                    checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"
                )

            # Save Checkpoint & Copy Latest to `latest-checkpoint.pt`
            torch.save({"model": model_state_dicts}, checkpoint_path)
            shutil.copy(checkpoint_path, checkpoint_dir / "latest-checkpoint.pt")

    def run_setup(self, n_train_examples: int) -> None:
        self.deepspeed_config = OmegaConf.structured(DefaultDeepSpeedConfig)
        
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=OmegaConf.to_container(self.deepspeed_config, resolve=True),
            gradient_accumulation_steps=self.grad_accumulation_steps,
        )
        self.accelerator = Accelerator(
            deepspeed_plugin=deepspeed_plugin,
            dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
            gradient_accumulation_steps=self.grad_accumulation_steps,
            mixed_precision="bf16" if self.enable_mixed_precision_training else "no",
        )

        # Create Optimizer and LR Scheduler =>> note that most of the LR Schedulers we use require `max_steps/epochs`
        #   => Optimizer should only operate on parameters that are *unfrozen* / trainable!
        n_train_examples = math.ceil(n_train_examples / self.global_batch_size) * self.global_batch_size
        if self.max_steps is None:
            num_training_steps = (n_train_examples * self.epochs) // self.global_batch_size
        else:
            num_training_steps = self.max_steps

        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            # Default AdamW w/ specified LR & Linear Warmup / Cosine Decay & Weight Decay
            #   => Create Parameter Groups --> bias terms, normalization layer parameters shouldn't be decayed!
            decay, no_decay = [], []
            for name, param in self.vla.named_parameters():
                if not param.requires_grad:
                    continue

                # Check on any parameters with fewer than 2 dimensions or with "bias" in the name
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)

            # Build Parameter Groups
            groups = [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

            # Create Optimizer & LR Scheduler
            self.optimizer = AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0
                
        elif self.lr_scheduler_type == "constant":
            num_warmup_steps = 0

            # Default AdamW w/ specified LR & Linear Warmup / Cosine Decay & Weight Decay
            #   => Create Parameter Groups --> bias terms, normalization layer parameters shouldn't be decayed!
            decay, no_decay = [], []
            for name, param in self.vla.named_parameters():
                if not param.requires_grad:
                    continue

                # Check on any parameters with fewer than 2 dimensions or with "bias" in the name
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)

            # Build Parameter Groups
            groups = [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

            # Create Optimizer & LR Scheduler
            self.optimizer = AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_constant_schedule(self.optimizer)
            
        else:
            raise ValueError(f"Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!")

        # Finalize Setup =>> Log!
        overwatch.info(
            "DeepSpeed Training Strategy =>> Finalized Training Setup:\n"
            f"         |-> Global (Effective) Batch Size = {self.global_batch_size}\n"
            f"         |-> Per-Device Batch Size = {self.per_device_batch_size}\n"
            f"         |-> Distributed World Size = {overwatch.world_size()}\n"
            f"         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n"
            f"         |-> LLM Backbone Gradient Checkpointing = {self.enable_gradient_checkpointing}\n"
            f"         |-> Use Mixed Precision = {self.enable_mixed_precision_training}\n"
            f"         |-> Default AdamW LR = {self.learning_rate}\n"
            f"         |-> AdamW Weight Decay = {self.weight_decay}\n"
            f"         |-> LR Scheduler Type = {self.lr_scheduler_type}\n"
            f"         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n"
            f"         |-> Dataset Size = {n_train_examples} Examples\n"
            f"         |-> Max Steps = {num_training_steps}\n"
        )

    def clip_grad_norm(self) -> None:
        self.accelerator.clip_grad_norm_(self.vla.parameters(), self.max_grad_norm)
        
    def run_training(
        self,
        dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        metrics: VLAMetrics,
        save_step: int = 2000,
        planning_loss_weight: float = 0.0,
        action_loss_weight: float = 1.0,
    ) -> None:
        # TODO: MiniVLA says "VLA training does not support gradient accumulation!", check why https://github.com/Stanford-ILIAD/openvla-mini/blob/main/prismatic/training/strategies/base_strategy.py#L269C51-L269C105
        
        epoch = 0
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        self.vla, self.optimizer, self.lr_scheduler, dataloader = self.accelerator.prepare(self.vla, self.optimizer, self.lr_scheduler, dataloader)

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vla.train()
            if self.vla.cfg.paradigm == "hierarchical":
                self.vla.vlm.vision_backbone.requires_grad_(False)
                self.vla.vlm.llm_backbone.requires_grad_(False)
                self.vla.vlm.projector.requires_grad_(False)
                self.vla.action_head.qw25_model.requires_grad_(False)

            # Zero-Gradients (just in case)
            self.optimizer.zero_grad()

            # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
            #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
            for batch in dataloader:
                with torch.autocast(
                    "cuda",
                    dtype=self.mixed_precision_dtype,
                    enabled=self.enable_mixed_precision_training,
                ):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    predicted_actions, planning_losses, action_loss = self.vla(batch)
                    assert isinstance(planning_losses, dict), "Expected `planning_losses` to be a dict!"
                    
                    language_planning_loss = planning_losses["language_planning_loss"]
                    visual_planning_loss = planning_losses["visual_planning_loss"]
                    image_foresight_planning_loss = planning_losses["image_foresight_planning_loss"]

                # Commit Loss (Prior to Gradient Accumulation Normalization)
                metrics.commit(
                    language_planning_loss=self.accelerator.gather(language_planning_loss).mean() if "language_planning" in self.vla.cfg.planning_heads and self.vla.cfg.paradigm != "hierarchical" else torch.tensor([0.0], device=self.accelerator.device),
                    visual_planning_loss=self.accelerator.gather(visual_planning_loss).mean() if "visual_planning" in self.vla.cfg.planning_heads and self.vla.cfg.paradigm != "hierarchical" else torch.tensor([0.0], device=self.accelerator.device),
                    image_foresight_planning_loss=self.accelerator.gather(image_foresight_planning_loss).mean() if "image_foresight_planning" in self.vla.cfg.planning_heads and self.vla.cfg.paradigm != "hierarchical" else torch.tensor([0.0], device=self.accelerator.device),
                )
                metrics.commit(action_loss=self.accelerator.gather(action_loss).mean())
                
                loss = planning_loss_weight * (language_planning_loss + visual_planning_loss + image_foresight_planning_loss) + action_loss_weight * action_loss
                normalized_loss = loss / self.grad_accumulation_steps
                self.accelerator.backward(normalized_loss)

                # Step =>> Only if Done w/ Gradient Accumulation
                if (metrics.global_step + 1) % self.grad_accumulation_steps == 0:
                    metrics.commit(update_step_time=True)

                    # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                    self.clip_grad_norm()

                    # Optimizer & LR Scheduler Step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    # Push Metrics
                    metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0], epoch=epoch)
                    status = metrics.push()

                    # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                    if self.max_steps is not None and metrics.global_step >= self.max_steps:
                        self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                        dist.barrier()

                        return

                    # Update Progress Bar
                    progress.update()
                    progress.set_description(status)


                # Compute epoch value using number of completed gradient steps
                previous_epoch = epoch
                epoch = (metrics.global_step + 1) // (len(dataset) // self.global_batch_size)
                # Check for Save Interval or Max Steps & Save Checkpoint
                # since there is no terminate sign for iteratable Dataset, we need to check the global step manually
                if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or epoch > previous_epoch or metrics.global_step == 1 or (metrics.global_step + 1) % save_step == 0:
                    self.save_checkpoint(
                        metrics.run_dir, metrics.global_step, epoch, self.accelerator.gather(loss).mean().item(), only_trainable=False)
                    dist.barrier()
                    del loss
                    torch.cuda.empty_cache()

                    if terminate:
                        return