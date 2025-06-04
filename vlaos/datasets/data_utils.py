"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple, Any, Dict, Type, Callable, Optional, Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import random
import math

from PIL import Image
from transformers import PreTrainedTokenizerBase
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

from vlaos.models.llm.prompting import PromptBuilder
from vlaos.models.vision import ImageTransform
from vlaos.datasets.cot_utils import get_language_cot_tags_list, get_visual_cot_tags_list, abbreviate_tag
import torchvision.transforms.functional as F

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

def augment_goal_image(batch: torch.Tensor) -> torch.Tensor:
    """
    Apply a sequence of augmentations to a batch of images.
    Args:
        batch: torch.Tensor of shape [B, 3, 256, 256], dtype=torch.uint8, values in [0, 255]
    Returns:
        torch.Tensor of same shape and dtype with augmentations applied in the given order.
    """
    B, C, H, W = batch.shape
    out = torch.empty_like(batch)
    for i in range(B):
        img = batch[i].float() / 255.0  # Normalize to [0,1]
        
        # 1. Random Resized Crop
        scale = (0.9, 0.9)
        ratio = (1.0, 1.0)
        area = H * W * random.uniform(*scale)
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect = math.exp(random.uniform(*log_ratio))
        w = int(round(math.sqrt(area * aspect)))
        h = int(round(math.sqrt(area / aspect)))
        top = random.randint(0, H - h)
        left = random.randint(0, W - w)
        img = F.resized_crop(img, top, left, h, w, (H, W))
        
        # 2. Random Brightness
        brightness_factor = random.uniform(1 - 0.2, 1 + 0.2)
        img = F.adjust_brightness(img, brightness_factor)
        
        # 3. Random Contrast
        contrast_factor = random.uniform(0.8, 1.2)
        img = F.adjust_contrast(img, contrast_factor)
        
        # 4. Random Saturation
        saturation_factor = random.uniform(0.8, 1.2)
        img = F.adjust_saturation(img, saturation_factor)
        
        # 5. Random Hue
        hue_factor = random.uniform(-0.05, 0.05)
        img = F.adjust_hue(img, hue_factor)
        
        # Convert back to uint8
        img = (img * 255.0).clamp(0, 255).to(torch.uint8)
        out[i] = img

    return out

def jitter(tags, sigma=4):
    new_tags = []
    for tag, val in tags:
        if tag in ["VISIBLE OBJECTS:", "GRIPPER POSITION:", "VISUAL OBJECT BBOXES:", "VISUAL FLOW:", "VISUAL AFFORDANCE:"]:
            coords = list(map(int, re.findall(r'\d+', val)))
            noisy = [min(max(0, int(c + np.random.normal(0, sigma))), 1023) for c in coords]
            new_tags.append((tag, "[" + ", ".join(map(str, noisy)) + "]"))
        else:
            new_tags.append((tag, val))
    return new_tags

def perturb_positions(channel, sigma=4.0):
    H, W = channel.shape
    new_ch = np.zeros_like(channel)
    
    # 获取所有值为1的坐标 (y, x)
    y_coords, x_coords = np.where(channel == 1)
    
    # 为每个坐标生成高斯噪声偏移量
    y_offsets = np.round(np.random.normal(0, sigma, len(y_coords))).astype(int)
    x_offsets = np.round(np.random.normal(0, sigma, len(x_coords))).astype(int)
    
    # 计算新坐标
    new_y = np.clip(y_coords + y_offsets, 0, H-1)
    new_x = np.clip(x_coords + x_offsets, 0, W-1)
    
    # 在新位置绘制点
    new_ch[new_y, new_x] = 1
    return new_ch

def augment_visual_planning_three_channels(image, sigma=4.0):
    augmented = np.zeros_like(image)
    for c in range(3):
        augmented[..., c] = perturb_positions(image[..., c], sigma)
    return augmented


def reasoning_dropout(reasoning: str, dropout_prob: float, planning_augmentation: bool = False, planning_aug_prob: float = 0.0):
    """Dropout reasoning tokens with probability `dropout_prob`."""
    if len(reasoning) == 0:
        return reasoning, "", ""

    reasoning_parts = reasoning.split("@")
    tags = [(reasoning_parts[i], reasoning_parts[i + 1]) for i in range(0, len(reasoning_parts), 2)]

    subset = np.random.rand(len(tags)) > dropout_prob

    subset_string = (
        "[" + ", ".join([abbreviate_tag(tag) for (tag, _), is_taken in zip(tags, subset) if is_taken]) + "]"
    )  # abbreviation

    excluded_tags = []

    if "EXCLUDE_TAGS" in os.environ:
        excluded_tags = os.environ["EXCLUDE_TAGS"].split(",")
        
    if planning_augmentation:
        # only for language planning because only for hierarchcial planning
        if random.random() < planning_aug_prob:
            # apply jitter to tags
            tags = jitter(tags)
        
    # parse to language planning and visual planning
    language_planning_tags = get_language_cot_tags_list()
    visual_planning_tags = get_visual_cot_tags_list()
    
    language_planning_reasoning = []
    visual_planning_reasoning = []
    
    for tag, is_taken in zip(tags, subset):
        if (is_taken and tag[0] not in excluded_tags):
            if tag[0] in language_planning_tags:
                language_planning_reasoning.append(f"{tag[0]} {tag[1]}")
            elif tag[0] in visual_planning_tags:
                visual_planning_reasoning.append(f"{tag[0]} {tag[1]}")
    
    if len(language_planning_reasoning) > 0:
        language_planning_reasoning = " ".join(language_planning_reasoning)
    else:
        language_planning_reasoning = None
        
    if len(visual_planning_reasoning) > 0:
        visual_planning_reasoning = " ".join(visual_planning_reasoning)
        visual_planning_reasoning = re.sub(r'(\d+)', r'<loc_\1>', visual_planning_reasoning)
    else:
        visual_planning_reasoning = None
        
    return language_planning_reasoning, visual_planning_reasoning, subset_string
    
def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}

@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32
    planning_mode: list = field(default_factory=list)  # "language_planning", "visual_planning", "image_foresight_planning"
    image_window_size: int = 2
    use_wrist_image: bool = True
    goal_img_augmentation: bool = False
    goal_img_augmentation_prob: float = 0.5

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        if "language_planning" in self.planning_mode:
            language_planning_input_ids, language_planning_labels = tuple(
                [instance[key] for instance in instances] for key in ("language_planning_input_ids", "language_planning_labels")
            )
        if "visual_planning" in self.planning_mode:
            visual_planning_input_ids, visual_planning_labels = tuple(
                [instance[key] for instance in instances] for key in ("visual_planning_input_ids", "visual_planning_labels")
            )

        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None
            
        if "actions" in instances[0]:
            actions = [instance["actions"] for instance in instances]
        else:
            actions = None

        if "proprios" in instances[0]:
            proprios = [instance["proprios"] for instance in instances]
        else:
            proprios = None
            
        if "depth_front" in instances[0]:
            depth_dict = {}
            rgb_dict = {}
            for instance in instances:
                for key in instance:
                    if "depth_" in key:
                        if key[6:] not in depth_dict:
                            depth_dict[key[6:]] = [instance[key]]
                        else: 
                            depth_dict[key[6:]].append(instance[key])
                    if "rgb_" in key:
                        if key[4:] not in rgb_dict:
                            rgb_dict[key[4:]] = [instance[key]]
                        else:
                            rgb_dict[key[4:]].append(instance[key])

            new_depth_dict = {}
            new_rgb_dict = {}
            for key in depth_dict:
                if key == "front":
                    new_key = "front_camera"
                elif key == "wrist":
                    new_key = "wrist_camera"
                elif key == "left":
                    new_key = "left_shoulder_camera"
                elif key == "right":
                    new_key = "right_shoulder_camera"
                new_depth_dict[new_key] = torch.stack(depth_dict[key])[:, -1, :, :, :]  # 3d encoder cannot take in history, so we only take the last timestep
                new_rgb_dict[new_key] = torch.stack(rgb_dict[key])[:, -1, :, :, :]  # 3d encoder cannot take in history, so we only take the last

            depth_dict = new_depth_dict
            rgb_dict = new_rgb_dict
        else:
            depth_dict = None
            rgb_dict = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        language_description = []
        for instance in instances:
            language_description.append(instance["language_description"])

        if "language_planning" in self.planning_mode:
            language_planning_input_ids = pad_sequence(language_planning_input_ids, batch_first=True, padding_value=self.pad_token_id)
            language_planning_labels = pad_sequence(language_planning_labels, batch_first=True, padding_value=IGNORE_INDEX)

            # Truncate (if necessary)
            language_planning_input_ids, language_planning_labels = language_planning_input_ids[:, : self.model_max_length], language_planning_labels[:, : self.model_max_length]

            # Get `attention_mask` by checking for `pad_token_id`
            language_planning_attention_mask = language_planning_input_ids.ne(self.pad_token_id)

            language_planning_reasoning = []
            for instance in instances:
                language_planning_reasoning.append(instance["language_planning_reasoning"])

        if "visual_planning" in self.planning_mode:
            visual_planning_input_ids = pad_sequence(visual_planning_input_ids, batch_first=True, padding_value=self.pad_token_id)
            visual_planning_labels = pad_sequence(visual_planning_labels, batch_first=True, padding_value=IGNORE_INDEX)
            
            # Truncate (if necessary)
            visual_planning_input_ids, visual_planning_labels = visual_planning_input_ids[:, : self.model_max_length], visual_planning_labels[:, : self.model_max_length]
            
            # Get `attention_mask` by checking for `pad_token_id`
            visual_planning_attention_mask = visual_planning_input_ids.ne(self.pad_token_id)

            visual_planning_reasoning = []
            for instance in instances:
                visual_planning_reasoning.append(instance["visual_planning_reasoning"])

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            language_description=language_description,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        if actions is not None:
            output["actions"] = torch.from_numpy(np.stack(actions)).to(input_ids.device)
        if proprios is not None:
            # padding to the same length for dexart
            for p_i in range(len(proprios)):
                if proprios[p_i].shape[1] == 32:
                    proprios[p_i] = torch.cat([proprios[p_i], torch.zeros((proprios[p_i].shape[0], 1)).to(proprios[p_i].device)], dim=1)  # add a zero column for proprioception
            output["proprios"] = torch.stack(proprios).to(input_ids.device)
        if depth_dict is not None:
            output["depth_dict"] = depth_dict
            output["rgb_dict"] = rgb_dict

        # for camera extrinsics
        extrinsics_keys = [
            key.replace("_extrinsics", "") for key in instances[0] if "extrinsics" in key
        ]
        if len(extrinsics_keys) > 0:
            extrinsics_dict = {}
            for key in extrinsics_keys:
                extrinsics_dict[key] = torch.stack([instance[key+"_extrinsics"] for instance in instances]).to(input_ids.device)
            output["extrinsics_dict"] = extrinsics_dict
            
        # for camera intrinsics
        intrinsics_keys = [
            key.replace("_intrinsics", "") for key in instances[0] if "intrinsics" in key
        ]
        if len(intrinsics_keys) > 0:
            intrinsics_dict = {}
            for key in intrinsics_keys:
                intrinsics_dict[key] = torch.stack([instance[key+"_intrinsics"] for instance in instances]).to(input_ids.device)
            output["intrinsics_dict"] = intrinsics_dict

        # original images
        output["original_imgs"] = [instance["original_imgs"] for instance in instances]
        output["original_imgs"] = torch.stack(output["original_imgs"]).to(input_ids.device)
        mean = torch.tensor([0.4850, 0.4560, 0.4060]).to(input_ids.device)
        std = torch.tensor([0.2290, 0.2240, 0.2250]).to(input_ids.device)
        # the normalization is for the cnn in the low-level action head
        output["original_imgs_without_normalization"] = output["original_imgs"].clone()
        output["original_imgs"] = (output["original_imgs"] - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

        if "language_planning" in self.planning_mode:
            output["language_planning_input_ids"] = language_planning_input_ids
            output["language_planning_attention_mask"] = language_planning_attention_mask
            output["language_planning_labels"] = language_planning_labels
            output["language_planning_reasoning"] = language_planning_reasoning
        
        if "visual_planning" in self.planning_mode:
            output["visual_planning_input_ids"] = visual_planning_input_ids
            output["visual_planning_attention_mask"] = visual_planning_attention_mask
            output["visual_planning_labels"] = visual_planning_labels
            output["visual_planning_reasoning"] = visual_planning_reasoning
            
        if "image_foresight_planning" in self.planning_mode:
            output["goal_image"] = torch.stack([instance["goal_image"] for instance in instances]).to(input_ids.device)
            if self.goal_img_augmentation and random.random() < self.goal_img_augmentation_prob:
                output["goal_image"] = augment_goal_image(output["goal_image"])
            output["goal_image"] = output["goal_image"].float().div(255/2).sub(1)
            
            # for low-level action head
            output["original_goal_img"] = torch.stack([instance["goal_image"] for instance in instances]).to(input_ids.device)
            output["original_goal_img"] = (output["original_goal_img"] - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

        return output

def prepare_input_ids(prompt_builder, conversation, print_prompt_limit, tokenizer, num_answer_tokens, predict_stop_token):
    for turn in conversation:
        prompt_builder.add_turn(turn["from"], turn["value"])

    if print_prompt_limit > 0:
        print("Conversation:", conversation)
        p = prompt_builder.get_prompt()
        print("Prompt:", p)

    # Tokenize (w/ `base_tokenizer`)
    input_ids = tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
    labels = list(input_ids)

    # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
    input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
    
    # critical, some tokenizers have different numbers of "end tokens".
    num_end_tokens = 1
    if isinstance(tokenizer, Qwen2TokenizerFast):
        # Qwen has <|im_end|><|endoftext|> for example
        num_end_tokens = 2

    # mask the input id tokens parts
    labels[0 : -(num_answer_tokens + num_end_tokens)] = IGNORE_INDEX
    if not predict_stop_token:
        labels[-num_end_tokens:] = IGNORE_INDEX

    return input_ids, labels

@dataclass
class RLDSBatchTransform:
    base_tokenizer: PreTrainedTokenizerBase
    visual_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    image_window_size: int = 1
    use_wrist_image: bool = False
    use_proprio: bool = True
    action_future_horizon: int = 0
    planning_mode: list = field(default_factory=list)  # "language_planning", "visual_planning", "image_foresight_planning"
    print_prompt_limit: int = 1
    reasoning_dropout_prob: float = 0.0
    load_camera_views: tuple = field(default_factory=tuple),
    load_depth: bool = False,
    planning_data_aug: bool = False
    
    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        
        assert len(self.planning_mode) == 0 or set(self.planning_mode).issubset({"language_planning", "visual_planning", "image_foresight_planning"}), f"Invalid planning mode: {set(self.planning_mode) - {'language_planning', 'visual_planning', 'image_foresight_planning'}}"
        
        if len(self.planning_mode) == 0:
            language_reasoning, visual_reasoning = None, None
        elif "language_planning" in self.planning_mode or "visual_planning" in self.planning_mode:   # have reasoning
            if self.planning_data_aug:
                language_reasoning, visual_reasoning, subset = reasoning_dropout(rlds_batch["reasoning"].decode(), dropout_prob=0.2, planning_augmentation=self.planning_data_aug, planning_aug_prob=0.5)
            else:
                language_reasoning, visual_reasoning, subset = reasoning_dropout(rlds_batch["reasoning"].decode(), dropout_prob=0.0, planning_augmentation=self.planning_data_aug)
        
        if self.use_proprio:
            proprio = rlds_batch["observation"]["proprio"]
            proprio = torch.tensor(proprio, dtype=torch.float32)
            assert proprio.shape[0] == self.image_window_size, f"Proprio shape {proprio.shape} does not match image window size {self.image_window_size}"

            # for camera extrinsics
            extrinsics_keys = [
                key for key in rlds_batch if "extrinsics" in key
            ]
            extrinsics = {
                key: torch.tensor(rlds_batch[key], dtype=torch.float32) for key in extrinsics_keys
            }

            # for camera intrinsics
            intrinsics_keys = [
                key for key in rlds_batch if "intrinsics" in key
            ]
            intrinsics = {
                key: torch.tensor(rlds_batch[key], dtype=torch.float32) for key in intrinsics_keys
            }

        # either a single or multi image, depending on image_window_size
        if self.image_window_size == 1:
            if len(self.load_camera_views) > 1:
                img = [Image.fromarray(rlds_batch["observation"]["image_"+key][0]) for key in self.load_camera_views]
                if self.load_depth:
                    rgb_dict = {
                        key: torch.from_numpy(rlds_batch["observation"]["image_"+key][0]) for key in self.load_camera_views
                    }
                    depth_dict = {
                        key: torch.from_numpy(rlds_batch["observation"]["depth_"+key][0]) for key in self.load_camera_views
                    }
            else:            
                img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
                if self.use_wrist_image:
                    img = [img, Image.fromarray(rlds_batch["observation"]["image_wrist"][0])]
        else:
            if len(self.load_camera_views) > 1:
                img = []
                if self.load_depth:
                    depth_dict = {key: [] for key in self.load_camera_views}
                    rgb_dict = {key: [] for key in self.load_camera_views}
                for t in range(self.image_window_size):
                    for key in self.load_camera_views:
                        img.append(Image.fromarray(rlds_batch["observation"]["image_"+key][t]))
                        if self.load_depth:
                            depth_dict[key].append(torch.from_numpy(rlds_batch["observation"]["depth_"+key][t]))
                            rgb_dict[key].append(torch.from_numpy(rlds_batch["observation"]["image_"+key][t]))
            else:
                img = [Image.fromarray(rlds_batch["observation"]["image_primary"][t]) for t in range(self.image_window_size)]
                if self.use_wrist_image:
                    # wrist images are interleaved
                    wrist_img = [
                        Image.fromarray(rlds_batch["observation"]["image_wrist"][t]) for t in range(self.image_window_size)
                    ]
                    img = [val for tup in zip(img, wrist_img) for val in tup]

        pixel_values = self.image_transform(img)
        
        if "image_foresight_planning" in self.planning_mode:
            if "image_primary" in rlds_batch["goal_image"]:
                goal_image = rlds_batch["goal_image"]["image_primary"]
            elif "image_front" in rlds_batch["goal_image"]:
                goal_image = rlds_batch["goal_image"]["image_front"]
            else:
                raise ValueError("Invalid goal image key in rlds_batch")

        vlm_conversation = []
        language_planning_head_conversation = []
        visual_planning_head_conversation = []

        # if there is no action horizon, remove it here.
        if self.action_future_horizon == 0:
            action = action[-1:]
        else:
            # TODO: check if this is correct: why last frames = future frames
            # get the last FH + 1 actions (current action + future ones) if required
            action = action[-self.action_future_horizon-1:]

        vlm_conversation.extend(
            [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
            ]
        )
        vlm_num_answer_tokens = 0

        if "language_planning" in self.planning_mode:
            if language_reasoning is None:
                # in case all tags are dropped out. only happen for hierarchical low level training
                language_reasoning = " ".join(get_language_cot_tags_list())
            language_planning_head_conversation.extend(
                [
                    {"from": "human", "value": f"What action should the robot take to {lang}?"},
                    {"from": "gpt", "value": f"{language_reasoning}"},
                ]
            )
            language_planning_head_num_answer_tokens = len(self.base_tokenizer(language_reasoning).input_ids)

        if "visual_planning" in self.planning_mode:
            if visual_reasoning is None:
                # in case all tags are dropped out. only happen for hierarchical low level training
                visual_reasoning = " ".join(get_visual_cot_tags_list())            
            visual_planning_head_conversation.extend(
                [
                    {"from": "human", "value": f"What action should the robot take to {lang}?"},
                    {"from": "gpt", "value": f"{visual_reasoning}"},
                ]
            )
            visual_planning_head_num_answer_tokens = len(self.visual_tokenizer(visual_reasoning).input_ids)

        # Construct Chat-based Prompt
        # prompt_builder = self.prompt_builder_fn("openvla")
        prompt_builder = self.prompt_builder_fn("qwen")

        # === VLM ===
        vlm_input_ids = None
        if len(self.planning_mode) == 0 or (len(self.planning_mode) == 1 and "image_foresight_planning" in self.planning_mode):
            vlm_input_ids, vlm_labels = prepare_input_ids(prompt_builder, vlm_conversation, self.print_prompt_limit, self.base_tokenizer, vlm_num_answer_tokens, self.predict_stop_token)
            self.print_prompt_limit -= 1
        if "language_planning" in self.planning_mode:
            # we do not print the prompt for the vlm when using language planning
            vlm_input_ids, vlm_labels = prepare_input_ids(prompt_builder, vlm_conversation, 0, self.base_tokenizer, vlm_num_answer_tokens, self.predict_stop_token)

            language_planning_prompt_builder = self.prompt_builder_fn("qwen")
            language_planning_input_ids, language_planning_labels = prepare_input_ids(language_planning_prompt_builder, language_planning_head_conversation, self.print_prompt_limit, self.base_tokenizer, language_planning_head_num_answer_tokens, self.predict_stop_token)
            self.print_prompt_limit -= 1
        if "visual_planning" in self.planning_mode:
            if vlm_input_ids is None:
                # we do not print the prompt for the vlm when using visual planning
                vlm_input_ids, vlm_labels = prepare_input_ids(prompt_builder, vlm_conversation, 0, self.base_tokenizer, vlm_num_answer_tokens, self.predict_stop_token)

            visual_planning_prompt_builder = self.prompt_builder_fn("qwen")
            visual_planning_input_ids, visual_planning_labels = prepare_input_ids(visual_planning_prompt_builder, visual_planning_head_conversation, self.print_prompt_limit, self.visual_tokenizer, visual_planning_head_num_answer_tokens, self.predict_stop_token)
            self.print_prompt_limit -= 1
        
        output = dict(
            pixel_values=pixel_values, 
            input_ids=vlm_input_ids, 
            labels=vlm_labels, 
            dataset_name=dataset_name, 
            proprios=proprio, 
            actions=action,
            file_names=rlds_batch["observation"]["file_name"],
            time_steps=rlds_batch["observation"]["timestep"],
            language_description=lang
        )
        
        if len(extrinsics) > 0:
            for key in extrinsics.keys():
                output[key] = extrinsics[key]
        if len(intrinsics) > 0:
            for key in intrinsics.keys():
                output[key] = intrinsics[key]        
        
        if self.load_depth:
            for key in self.load_camera_views:
                output["depth_"+key.replace("primary", "front")] = torch.stack(depth_dict[key])
                output["rgb_"+key.replace("primary", "front")] = torch.stack(rgb_dict[key])

        if "language_planning" in self.planning_mode:
            output["language_planning_input_ids"] = language_planning_input_ids
            output["language_planning_labels"] = language_planning_labels
            output["language_planning_reasoning"] = language_reasoning
        
        if "visual_planning" in self.planning_mode:
            output["visual_planning_input_ids"] = visual_planning_input_ids
            output["visual_planning_labels"] = visual_planning_labels
            output["visual_planning_reasoning"] = visual_reasoning
        
        if "image_foresight_planning" in self.planning_mode:
            output["goal_image"] = torch.from_numpy(goal_image).permute(2, 0, 1)

        # also, load original images
        if self.image_window_size == 1:
            if len(self.load_camera_views) > 1:
                original_imgs = [Image.fromarray(rlds_batch["observation"]["image_"+key][0]) for key in self.load_camera_views]
                original_imgs = torch.from_numpy(np.stack(original_imgs))
            else:
                original_imgs = rlds_batch["observation"]["image_primary"][0]
                if self.use_wrist_image:
                    original_imgs = [original_imgs, rlds_batch["observation"]["image_wrist"][0]]
                    original_imgs = torch.from_numpy(np.stack(original_imgs))
                else:
                    original_imgs = torch.from_numpy(original_imgs).unsqueeze(0)
        else:
            if len(self.load_camera_views) > 1:
                original_imgs = []
                for t in range(self.image_window_size):
                    for key in self.load_camera_views:
                        original_imgs.append(Image.fromarray(rlds_batch["observation"]["image_"+key][t]))
            else:
                original_imgs = [rlds_batch["observation"]["image_primary"][t] for t in range(self.image_window_size)]
                if self.use_wrist_image:
                    # wrist images are interleaved
                    wrist_img = [
                        rlds_batch["observation"]["image_wrist"][t] for t in range(self.image_window_size)
                    ]
                    original_imgs = [val for tup in zip(original_imgs, wrist_img) for val in tup]
                
            original_imgs = torch.from_numpy(np.stack(original_imgs))
        output["original_imgs"] = original_imgs.permute(0, 3, 1, 2).float()

        return output