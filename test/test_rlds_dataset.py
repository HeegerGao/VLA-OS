import sys
sys.path.append(".")
sys.path.append("..")
from pathlib import Path
from vlaos.datasets.materialize import get_continuous_vla_dataset_and_collator
from vlaos.models import Qwen25LLMBackbone, DinoSigLIPViTBackbone, QwenVLM
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from utils.robot_utils import set_seed_everywhere
from copy import deepcopy
import time

set_seed_everywhere(42)
image_window_size = 2

# for libero, deformable
use_wrist_image = True

# for colosseum
# given_camera_views=["primary", "wrist", "left", "right"]  # for colosseum

# for dexart
# given_camera_views=["primary"]    # for dexart
# use_wrist_image = False

# for furniturebench
# given_camera_views=["front", "rear", "wrist"]    # for furniturebench

# for peract2
# given_camera_views=["primary", "over_shoulder_left", "wrist_left", "over_shoulder_right", "overhead", "over_shoulder_right", "wrist_right"]    # for peract2


# for libero, deformable
image_sequence_len = image_window_size if not use_wrist_image else image_window_size * 2

# for colosseum, dexart, furniturebench, peract2
# image_sequence_len = len(given_camera_views) * image_window_size

vision_backbone = DinoSigLIPViTBackbone("dinosiglip-vit-so-224px", "resize-naive", default_image_size=224, image_sequence_len=image_sequence_len, pretrained=False)
image_transform = vision_backbone.get_image_transform()
llm_backbone = Qwen25LLMBackbone(
    "qwen25-0_5b",
    hf_token=".hf_token",
    inference_mode=False,
    pretrained=False,
)
tokenizer = llm_backbone.get_tokenizer()

visual_planning_tokenizer = deepcopy(tokenizer)
visual_planning_tokenizer.add_tokens([f"<loc_{i}>" for i in range(1024)])


vla_dataset, collator = get_continuous_vla_dataset_and_collator(
    data_root_dir=Path('dataset/libero'),
    data_mix='libero_10',
    # data_root_dir=Path('dataset/deformable'),
    # data_mix='deformable',
    # data_root_dir=Path('dataset/dexart'),
    # data_mix='dexart',    
    # data_root_dir=Path('dataset/colosseum'),
    # data_mix='colosseum',
    # data_root_dir=Path('dataset/libero'),
    # data_mix='libero_90',
    # data_root_dir=Path('dataset/furniturebench'),
    # data_mix='furniturebench',
    # data_root_dir=Path('dataset/peract2'),
    # data_mix='peract2',    
    image_transform=image_transform,
    tokenizer=tokenizer,
    prompt_builder_fn=llm_backbone.prompt_builder_fn,
    default_image_resolution=(3, 224, 224),
    image_aug = False,
    future_action_window_size = 7,
    goal_image_step = 7,
    image_window_size = image_window_size,
    use_wrist_image = use_wrist_image,
    use_proprio = True,
    planning_mode = ["language_planning", "visual_planning"],
    # planning_mode = [],
    visual_tokenizer=visual_planning_tokenizer,
    # given_camera_views=given_camera_views,    # for colosseum, dexart, furniturebench, peract2
    # load_depth=True,  # for colosseum, peract2
    load_depth=False,   # for libero, deformable, furniturebench, dexart
    shuffle_buffer_size=8000,
    # sample_fraction=0.1,  # for the data scaling experiments
    # planning_data_augmentation=True,
)

dataloader = DataLoader(
    vla_dataset,
    batch_size=4,
    sampler=None,
    collate_fn=collator,
    num_workers=0,
)

step = 0
start_time = time.time()
for batch in dataloader:
    # print(batch.keys())
    now_time = time.time()
    print(f"step: {step}", "time:", now_time - start_time)
    start_time = now_time
    step += 1
