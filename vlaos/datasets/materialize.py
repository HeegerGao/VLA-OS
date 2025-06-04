"""
materialize.py

Factory class for initializing pretraining datasets on a per-VLM basis; provides and exports individual functions for
clear control flow.
"""

import os
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from pathlib import Path

from vlaos.models.llm.prompting import PromptBuilder
from vlaos.models.vision import ImageTransform, WrapSequenceImageTransform
from vlaos.datasets.datasets import AlignDataset, FinetuneDataset, RLDSDataset
from vlaos.datasets.data_utils import PaddedCollatorForLanguageModeling, PaddedCollatorForActionPrediction, RLDSBatchTransform

# Dataset Initializers =>> Maps Stage --> cls()
DATASET_INITIALIZER = {"align": AlignDataset, "finetune": FinetuneDataset, "full-finetune": FinetuneDataset}

def get_dataset_and_collator(
    stage: str,
    cfg,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
) -> Tuple[Dataset, PaddedCollatorForLanguageModeling]:
    dataset_cls = DATASET_INITIALIZER[stage]
    collator = PaddedCollatorForLanguageModeling(
        tokenizer.model_max_length, tokenizer.pad_token_id, default_image_resolution, padding_side=padding_side
    )

    # Switch on `stage`
    if stage == "align":
        annotation_json, image_dir = cfg.align_stage_components
        dataset = dataset_cls(
            Path(cfg.dataset_root_dir) / annotation_json, cfg.dataset_root_dir / image_dir, image_transform, tokenizer
        )
        return dataset, collator

    elif stage == "finetune":
        annotation_json, image_dir = cfg.finetune_stage_components
        dataset = dataset_cls(
            Path(cfg.dataset_root_dir) / annotation_json,
            Path(cfg.dataset_root_dir) / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator

    elif stage == "full-finetune":
        annotation_json, image_dir = cfg.finetune_stage_components
        dataset = dataset_cls(
            Path(cfg.dataset_root_dir) / annotation_json,
            Path(cfg.dataset_root_dir) / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator

    else:
        raise ValueError(f"Stage `{stage}` is not supported!")

# we use continuous actions so we do not return action tokenizer
def get_continuous_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 256_000,
    train: bool = True,
    image_aug: bool = False,
    future_action_window_size: int = 0,
    goal_image_step: int = 7,
    image_window_size: int = 1, # images in the sequence of a single view
    use_wrist_image: bool = True,
    use_proprio = True,
    planning_mode: list = [],  # "no_planning", "language_planning", "visual_planning", "multi_modal_planning"
    visual_tokenizer = None,
    given_camera_views = None,
    load_depth: bool = False,
    planning_data_augmentation: bool = False,
    sample_fraction: float = 1.0,
) -> Tuple[Dataset, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""

    if given_camera_views is None:
        load_camera_views = ("primary", "wrist") if use_wrist_image else ("primary",)
    else:
        load_camera_views = tuple(given_camera_views)
    
    batch_transform = RLDSBatchTransform(
        tokenizer,
        visual_tokenizer,
        image_transform,
        prompt_builder_fn,
        predict_stop_token=predict_stop_token,
        image_window_size=image_window_size,
        use_wrist_image=use_wrist_image,
        load_camera_views=load_camera_views,
        load_depth=load_depth,
        use_proprio=use_proprio,
        action_future_horizon=future_action_window_size,
        planning_mode=planning_mode,
        print_prompt_limit = 1 if len(planning_mode) <= 1 else len(planning_mode),
        planning_data_aug=planning_data_augmentation,
    )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, 
        tokenizer.pad_token_id, 
        padding_side=padding_side, 
        planning_mode=planning_mode,
        image_window_size=image_window_size,
        use_wrist_image=use_wrist_image,
        goal_img_augmentation=planning_data_augmentation,
    )

    # Build RLDS Iterable Dataset
    dataset = RLDSDataset(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
        future_action_window_size=future_action_window_size,
        goal_image_step=goal_image_step,
        image_window_size=image_window_size,
        load_camera_views=load_camera_views,
        load_depth=load_depth,
        load_reasoning=True if len(planning_mode) > 0 else False,
        sample_fraction=sample_fraction,
    )

    return dataset, collator