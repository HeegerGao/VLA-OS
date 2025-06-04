import os
from pathlib import Path
import torch
import torch.distributed as dist
from utils.utils import set_global_seed
from omegaconf import OmegaConf

from vlaos.overwatch import initialize_overwatch

from vlaos.datasets import get_dataset_and_collator
from vlaos.training import FSDPStrategy
from vlaos.training.metrics import Metrics
from utils.model_utils import load_qwen_vlm

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

def pretrain(cfg):
    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    
    # --------------------------------------- Initialization -----------------------------------------------------
    torch.cuda.set_device(device_id := (overwatch.local_rank()))
    torch.cuda.empty_cache()
    
    model_id = cfg.model.model_id
    if (dataset_id := cfg.dataset.dataset_id) == "llava-v15":
        cfg.run_id = f"{model_id}+stage-{cfg.training.stage}+x{cfg.seed}"
    else:
        cfg.run_id = f"{dataset_id}+{model_id}+stage-{cfg.training.stage}+x{cfg.seed}"

    # Start =>> Build Directories and Set Randomness
    hf_token = cfg.hf_token
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (Path(cfg.run_root_dir) / cfg.run_id), exist_ok=True)
    os.makedirs(Path(cfg.run_root_dir) / cfg.run_id / "checkpoints", exist_ok=True)
    if overwatch.is_rank_zero():
        OmegaConf.save(cfg, run_dir / "config.yaml")
            
    # --------------------------------------- Model -----------------------------------------------------

    vlm, image_transform, tokenizer = load_qwen_vlm(
        model_id=model_id,
        vision_backbone_id=cfg.model.vision_backbone_id,
        image_resize_strategy=cfg.model.image_resize_strategy,
        default_image_size=cfg.model.default_image_size,
        image_sequence_len=cfg.model.image_sequence_len,
        llm_backbone_id=cfg.model.llm_backbone_id,
        inference_mode=False,
        stage="freeze",
        pretrained_checkpoint=None,
        run_dir=run_dir,
    )

    # Load Weights from Checkpoint (depends on stage, config)
    if cfg.model.pretrained_checkpoint is not None:
        overwatch.info(f"Invoking `VLM.load_checkpoint()` for `{model_id}` => Training Stage: `{cfg.training.stage}`")
        vlm.load_from_checkpoint(cfg.training.stage, run_dir, pretrained_checkpoint=cfg.model.pretrained_checkpoint)

    # --------------------------------------- Dataset -----------------------------------------------------

    # Get Dataset for Specified Stage
    overwatch.info(f"Creating Dataset `{cfg.dataset.dataset_id}` => Stage: `{cfg.training.stage}`")
    train_dataset, collator = get_dataset_and_collator(
        cfg.training.stage,
        cfg.dataset,
        image_transform,
        tokenizer,
        prompt_builder_fn=vlm.llm_backbone.prompt_builder_fn,
        default_image_resolution=vlm.vision_backbone.default_image_resolution,
        padding_side=tokenizer.padding_side,
    )

    # --------------------------------------- Training -----------------------------------------------------

    train_strategy = FSDPStrategy(
            vlm=vlm,
            device_id=device_id,
            epochs=cfg.training.epochs,
            max_steps=cfg.training.max_steps,
            global_batch_size=cfg.training.global_batch_size,
            per_device_batch_size=cfg.training.per_device_batch_size,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            max_grad_norm=cfg.training.max_grad_norm,
            lr_scheduler_type=cfg.training.lr_scheduler_type,
            warmup_ratio=cfg.training.warmup_ratio,
            enable_gradient_checkpointing=cfg.training.enable_gradient_checkpointing,
            enable_mixed_precision_training=cfg.training.enable_mixed_precision_training,
            reduce_in_full_precision=cfg.training.reduce_in_full_precision,
            worker_init_fn=worker_init_fn,
            sharding_strategy=cfg.training.sharding_strategy,    # "shard-grad-op" or "full-shard"
        )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(train_dataset))

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info("Creating Metrics with Active Trackers => `jsonl, wandb`")
    metrics = Metrics(
        ("jsonl", "wandb"),
        cfg.run_id,
        run_dir,
        OmegaConf.to_container(cfg),
        cfg.training.stage,
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        grad_accumulation_steps=train_strategy.grad_accumulation_steps,
    )

    # Run Training
    overwatch.info("Starting Training Loop")
    train_strategy.run_training(train_dataset, collator, metrics, stage=cfg.training.stage, seed=cfg.seed)

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    cfg = OmegaConf.load("config/train_vlm.yaml")
    pretrain(cfg)