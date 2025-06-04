import json
import os
import re
from pathlib import Path
from omegaconf import OmegaConf

import torch
import torch.distributed as dist

from vlaos.overwatch import initialize_overwatch
from utils.model_utils import load_qwen_vlm
from utils.utils import set_global_seed

from vlaos.models import ActionOnlyVLA, IntegratedVLA, HierarchicalVLA
from vlaos.datasets import get_continuous_vla_dataset_and_collator
from vlaos.training import AcceleratorStrategy
from vlaos.training.metrics import VLAMetrics
from vlaos.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

def train(cfg) -> None:
    overwatch.info("VLA Training ...")

    # Configure Unique Run Name & Save Directory
    cfg.run_id = f"{cfg.vla.paradigm}+{cfg.vla.vla_id}+stage-{cfg.training.stage}+x{cfg.seed}+{cfg.training.training_algo}"
    if cfg.data.image_aug:
        cfg.run_id += "--image_aug"
    if cfg.vla.paradigm == "integrated":
        for head in cfg.vla.planning_heads:
            cfg.run_id += f"+{head}"
        cfg.run_id += f"+{cfg.vla.planning_mode}"
    if cfg.vla.paradigm == "hierarchical":
        for head in cfg.vla.planning_heads:
            cfg.run_id += f"+{head}"

    # Start =>> Build Directories and Set Randomness
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (Path(cfg.run_root_dir) / cfg.run_id), exist_ok=True)
    os.makedirs(Path(cfg.run_root_dir) / cfg.run_id / "checkpoints", exist_ok=True)
    if overwatch.is_rank_zero():
        OmegaConf.save(cfg, run_dir / "config.yaml")

    # Load VLA checkpoint (if resuming from training) or Base VLM otherwise (from `cfg.vla.base_vlm` ID or Path)
    #   =>> Note :: Verifies that all parameters are loaded in FP32 on load!
    
    vlm, image_transform, tokenizer = load_qwen_vlm(
        model_id=cfg.vla.vlm.model_id,
        vision_backbone_id=cfg.vla.vlm.vision_backbone_id,
        image_resize_strategy=cfg.vla.vlm.image_resize_strategy,
        default_image_size=cfg.vla.vlm.default_image_size,
        image_sequence_len=(cfg.data.image_window_size if not cfg.data.use_wrist_image else cfg.data.image_window_size * 2) if cfg.data.given_camera_views is None else len(cfg.data.given_camera_views) * cfg.data.image_window_size,
        llm_backbone_id=cfg.vla.vlm.llm_backbone_id,
        inference_mode=False,
        stage=cfg.training.stage,
        pretrained_checkpoint=cfg.vla.vlm.pretrained_checkpoint,
        run_dir=run_dir,
        load_pretrained_weights_from_hf=False,
    )
    
    visual_tokenizer = None
    if cfg.vla.paradigm == "action-only":
        vla = ActionOnlyVLA(cfg.vla, vlm, training_algo=cfg.training.training_algo, skewed_timesteps=cfg.training.skewed_timesteps)
    elif cfg.vla.paradigm == "integrated":
        vla = IntegratedVLA(cfg.vla, vlm, training_algo=cfg.training.training_algo, skewed_timesteps=cfg.training.skewed_timesteps)
        if "visual_planning" in cfg.vla.planning_heads:
            visual_tokenizer = vla.visual_planning_tokenizer
    elif cfg.vla.paradigm == "hierarchical":
        vla = HierarchicalVLA(cfg.vla, vlm, training_algo=cfg.training.training_algo, skewed_timesteps=cfg.training.skewed_timesteps)
        if "visual_planning" in cfg.vla.planning_heads:
            visual_tokenizer = vla.visual_planning_tokenizer
    else:
        raise ValueError(f"Invalid VLA Paradigm: {cfg.vla.paradigm}")
    
    if cfg.vla.load_from_checkpoint:
        if cfg.vla.paradigm == "hierarchical":
            overwatch.info(f"Loading high-level checkpoint from {cfg.vla.high_level_checkpoint}")
            vla.load_from_checkpoint(high_level_checkpoint=cfg.vla.high_level_checkpoint)
            overwatch.info(f"Loading action head checkpoint from {cfg.vla.action_head_checkpoint}")
            vla.load_from_checkpoint(low_level_checkpoint=cfg.vla.action_head_checkpoint)
        else:
            overwatch.info(f"Loading VLA checkpoint from {cfg.vla.checkpoint}")
            vla.load_from_checkpoint(pretrained_checkpoint=cfg.vla.checkpoint)
    
    vlm.freeze_backbones(cfg.training.stage)
    if cfg.vla.paradigm == "hierarchical":
        vlm.freeze_backbone("freeze")
    
    # [Validate] Model should be in Full Precision!
    for param in vlm.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    # Print number of total/trainable model parameters
    num_params = sum(p.numel() for p in vla.parameters())
    num_trainable_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    action_head_num_params = sum(p.numel() for p in vla.action_head.parameters())
    
    language_planning_head_num_params = 0
    visual_planning_head_num_params = 0
    image_foresight_planning_head_num_params = 0
    if cfg.vla.paradigm == "integrated" or cfg.vla.paradigm == "hierarchical":
        if "language_planning" in cfg.vla.planning_heads:
            language_planning_head_num_params = sum(p.numel() for p in vla.language_planning_head.parameters())
        if "visual_planning" in cfg.vla.planning_heads:
            visual_planning_head_num_params = sum(p.numel() for p in vla.visual_planning_head.parameters())
        if "image_foresight_planning" in cfg.vla.planning_heads:
            image_foresight_planning_head_num_params = sum(p.numel() for p in vla.image_foresight_head.parameters())
    
    overwatch.info(
        f"# Parameters of VLA (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable, {action_head_num_params / 10**6:.3f} of which are in the Action Head, {language_planning_head_num_params / 10**6:.3f} in the Language Planning Head, {visual_planning_head_num_params / 10**6:.3f} in the Visual Planning Head, {image_foresight_planning_head_num_params / 10**6:.3f} in the Image Foresight Planning Head"
    )

    # Get VLA Dataset & Collator
    overwatch.info(f"Creating VLA Dataset with Mixture `{cfg.data.data_mix}`")
    vla_dataset, collator = get_continuous_vla_dataset_and_collator(
        cfg.data.root_dir,
        cfg.data.data_mix,
        image_transform=image_transform,
        tokenizer=tokenizer,
        prompt_builder_fn=vlm.llm_backbone.prompt_builder_fn,
        default_image_resolution=vlm.vision_backbone.default_image_resolution,
        shuffle_buffer_size=cfg.data.shuffle_buffer_size,
        image_aug=cfg.data.image_aug,    # boolen
        future_action_window_size=cfg.data.future_action_window_size,
        image_window_size=cfg.data.image_window_size,
        use_wrist_image=cfg.data.use_wrist_image,  # will double the sequence length
        use_proprio=cfg.data.use_proprio,
        planning_mode=cfg.vla.planning_heads,
        visual_tokenizer=visual_tokenizer,
        goal_image_step=cfg.data.goal_image_step,
        given_camera_views=cfg.data.given_camera_views,
        load_depth=cfg.data.load_depth,
        sample_fraction=1.0 if "sample_fraction" not in cfg.data else cfg.data.sample_fraction,
    )

    # Save dataset statistics for de-normalization at inference time
    if overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)
        
    # Create Train Strategy
    overwatch.info("Initializing Train Strategy `AcceleratorStrategy`")
    train_strategy = AcceleratorStrategy(
        vla=vla,
        trainig_stage=cfg.training.stage,
        epochs=cfg.training.epochs,
        max_steps=cfg.training.max_steps,
        global_batch_size=cfg.training.global_batch_size,
        per_device_batch_size=cfg.training.per_device_batch_size,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(n_train_examples=len(vla_dataset))
    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info("Creating Metrics with Active Trackers => `jsonl, wandb`")
    metrics = VLAMetrics(
        ("jsonl", "wandb"),
        cfg.run_id,
        run_dir,
        OmegaConf.to_container(cfg),
        wandb_project=cfg.wandb_project,
        resume_step=cfg.resume_step,
        resume_epoch=cfg.resume_epoch,
        window_size=cfg.data.image_window_size,
        grad_accumulation_steps=train_strategy.grad_accumulation_steps,
    )

    # Run VLA Training
    overwatch.info("Starting VLA Training Loop")
    train_strategy.run_training(
        vla_dataset,
        collator,
        metrics,
        save_step=cfg.training.save_step,
        planning_loss_weight=cfg.training.planning_loss_weight,
        action_loss_weight=cfg.training.action_loss_weight,
    )

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg)