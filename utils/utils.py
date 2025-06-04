import torch
import random
import numpy as np
import os
from typing import Optional, Callable
import logging
from datetime import datetime
import re
from typing import List, Tuple
from difflib import get_close_matches
from transformers import AutoModel, CLIPImageProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

def set_global_seed(seed: int, get_worker_init_fn: bool = False) -> Optional[Callable[[int], None]]:
    """Sets seed for all randomness libraries (mostly random, numpy, torch) and produces a `worker_init_fn`"""
    assert np.iinfo(np.uint32).min < seed < np.iinfo(np.uint32).max, "Seed outside the np.uint32 bounds!"

    # Set Seed as an Environment Variable
    os.environ["EXPERIMENT_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return worker_init_function if get_worker_init_fn else None


def worker_init_function(worker_id: int) -> None:
    """
    Borrowed directly from PyTorch-Lightning; inspired by this issue comment in the PyTorch repo:
        > Ref: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562

    Intuition: You can think of the seed sequence spawn function as a "janky" torch.Generator() or jax.PRNGKey that
    you can run iterative splitting on to get new (predictable) randomness.

    :param worker_id: Identifier for the given worker [0, num_workers) for the Dataloader in question.
    """
    # Get current `rank` (if running distributed) and `process_seed`
    global_rank, process_seed = int(os.environ["LOCAL_RANK"]), torch.initial_seed()

    # Back out the "base" (original) seed - the per-worker seed is set in PyTorch:
    #   > https://pytorch.org/docs/stable/data.html#data-loading-randomness
    base_seed = process_seed - worker_id

    # "Magic" code --> basically creates a seed sequence that mixes different "sources" and seeds every library...
    seed_seq = np.random.SeedSequence([base_seed, worker_id, global_rank])

    # Use 128 bits (4 x 32-bit words) to represent seed --> generate_state(k) produces a `k` element array!
    np.random.seed(seed_seq.generate_state(4))

    # Spawn distinct child sequences for PyTorch (reseed) and stdlib random
    torch_seed_seq, random_seed_seq = seed_seq.spawn(2)

    # Torch Manual seed takes 64 bits (so just specify a dtype of uint64
    torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64)[0])

    # Use 128 Bits for `random`, but express as integer instead of as an array
    random_seed = (random_seed_seq.generate_state(2, dtype=np.uint64).astype(list) * [1 << 64, 1]).sum()
    random.seed(random_seed)



# === BFloat16 Support ===


def check_bloat16_supported() -> bool:
    try:
        import packaging.version
        import torch.cuda.nccl as nccl
        import torch.distributed as dist

        return (
            (torch.version.cuda is not None)
            and torch.cuda.is_bf16_supported()
            and (packaging.version.parse(torch.version.cuda).release >= (11, 0))
            and dist.is_nccl_available()
            and (nccl.version() >= (2, 10))
        )

    except Exception:
        return False


def setup_logger(log_dir: str, 
                 log_filename: str = None, 
                 terminal_level=logging.INFO, 
                 file_level=logging.DEBUG) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    if log_filename is None:
        log_filename = datetime.now().strftime("log_%Y%m%d_%H%M%S.txt")

    log_path = os.path.join(log_dir, log_filename)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Terminal handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(terminal_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def draw_bbox_from_topleft_and_bottomright(top_left_patch_idx, bottom_right_patch_idx, img_size=224, patch_size=7):
    """
    在一张背景全 0 的 224x224 灰度图上，根据从 0 开始编号的 patch 索引画出 bbox。

    参数：
        top_left_patch_idx (int): 左上角 patch 编号（从 0 开始, 共 32 列×32 行）。
        bottom_right_patch_idx (int): 右下角 patch 编号（同上）。
        img_size (int): 图像边长，默认 224。
        patch_size (int): 每个 patch 的像素大小，默认 7。

    返回：
        img (np.ndarray): 画好 bbox 的 224x224 uint8 图像，框线像素值为 255。
    """
    grid_n = img_size // patch_size  # patch 数：32

    # 0-based 编号直接计算 row 和 col
    col1 = top_left_patch_idx % grid_n
    row1 = top_left_patch_idx // grid_n
    col2 = bottom_right_patch_idx % grid_n
    row2 = bottom_right_patch_idx // grid_n

    # 转换为像素坐标
    y1 = row1 * patch_size
    x1 = col1 * patch_size
    y2 = (row2 + 1) * patch_size - 1
    x2 = (col2 + 1) * patch_size - 1

    # 创建全 0 背景
    img = np.zeros((img_size, img_size), dtype=np.uint8)

    # check if the bbox is out of range and replace it with the last patch
    if y1 < 0:
        y1 = 0
    if y2 >= img_size:
        y2 = img_size - 1
    if x1 < 0:
        x1 = 0
    if x2 >= img_size:
        x2 = img_size - 1

    # 上边框：从 y1 开始，向下 border_width 行
    img[y1 : y1 + patch_size, x1 : x2 + 1] = 255
    # 下边框：从 y2 向上 border_width 行
    img[y2 - patch_size + 1 : y2 + 1, x1 : x2 + 1] = 255
    # 左边框：从 x1 开始，向右 border_width 列
    img[y1 : y2 + 1, x1 : x1 + patch_size] = 255
    # 右边框：从 x2 向左 border_width 列
    img[y1 : y2 + 1, x2 - patch_size + 1 : x2 + 1] = 255

    return img

def draw_flow_from_flow_tokens(flow_tokens, img_size=224, patch_size=7):
    """
    在一张背景全 0 的 224×224 灰度图上，根据 0-based 编号的 patch 索引列表
    将每个对应的 patch 区域填充值为 fill_value。

    参数:
        patch_indices: 一系列 patch 编号（0-based，先“从上到下”后“从左到右”，共 32 列×32 行）。
        img_size:       图像边长（默认 224）。
        patch_size:     每个 patch 的边长（默认 7）。
        fill_value:     要填充的像素值（默认 1.0）。

    返回:
        img: 形状为 (img_size, img_size) 的 ndarray，dtype float32，patch 区域为 fill_value，其它为 0。
    """
    # 计算每行/列 patch 数
    grid_n = img_size // patch_size  # 32

    # 初始化全 0 图
    img = np.zeros((img_size, img_size), dtype=np.float32)

    for idx in flow_tokens:
        if idx < 0 or idx >= grid_n * grid_n:
            print(f"patch index {idx} out of range [0, {grid_n*grid_n-1}]")
            continue
        # 计算 patch 对应的行列（0-based）
        row = idx // grid_n
        col = idx % grid_n
        # 计算像素范围
        y1 = row * patch_size
        x1 = col * patch_size
        # 填充整个 patch 区域
        img[y1 : y1 + patch_size, x1 : x1 + patch_size] = 1

    return img


def robust_parse_locs(
    text: str,
    headers: List[str] = None,
    fuzzy_thresh: float = 0.6
) -> Tuple[List[int], List[Tuple[int,int]], List[int], List[int]]:
    """
    从一段可能有拼写/格式错误的文本中，尽量提取所有 loc，并尝试按照
      BBOXES → FLOW → AFFORDANCE
    三个区段来分组。返回：
      1) all_locs: 文本中出现过的所有 loc（按出现顺序）
      2) bbox_pairs: BBOXES 部分两两一组（如果是奇数，多余的最后一个丢弃）
      3) flow_locs: FLOW 部分（顺序不变）
      4) aff_locs: AFFORDANCE 部分（顺序不变）

    参数:
      text: 待处理的原始字符串
      headers: 可选自定义三段落标题，默认为 ["BBOXES", "FLOW", "AFFORDANCE"]
      fuzzy_thresh: 在找标题时，用 difflib 匹配的相似度阈值（0~1）
    """
    if headers is None:
        headers = ["BBOXES", "FLOW", "AFFORDANCE"]

    # 1. 抽出所有 loc 数字，连同它在文本中的位置
    #    容错：允许写成 <loc123>, loc_123, loc 123, LOC-123, … 取连续数字
    loc_pattern = re.compile(r"""
        <?\s*               # 可选尖括号和前空白
        [lL][oO0][cC]       # loc 忽大小写，O/0 容错
        [\s_\-]*            # 可选分隔符
        (\d{1,6})           # 捕获 1~6 位数字
        \s*>?               # 可选尾尖括号和空白
    """, re.VERBOSE)
    all_items = [(int(m.group(1)), m.start()) for m in loc_pattern.finditer(text)]
    all_locs = [v for v,_ in all_items]

    # 2. 在 text 里找三个段落分界点，用模糊匹配来定位即使拼错也找得到
    lower = text.lower()
    bounds = {}
    for hdr in headers:
        # 在原文里按单词拆，找最接近 hdr.lower() 的词
        words = re.findall(r"[A-Za-z]+", text)
        # 用 difflib 找最像 hdr 的词
        close = get_close_matches(hdr.lower(), [w.lower() for w in words], n=1, cutoff=fuzzy_thresh)
        if close:
            # 找到那个词在原文中的位置
            pat = re.compile(re.escape(close[0]), re.IGNORECASE)
            m = pat.search(text)
            bounds[hdr] = m.start()
        else:
            bounds[hdr] = None

    # 按出现顺序排列三个 header
    sorted_hdrs = sorted(
        [(hd, bounds[hd]) for hd in headers if bounds[hd] is not None],
        key=lambda x: x[1]
    )
    names = [hd for hd,_ in sorted_hdrs]
    poses = [pos for _,pos in sorted_hdrs]

    # 3. 根据各 loc 的起始位置，把它们分到三个区段
    segs = {"BBOXES":[], "FLOW":[], "AFFORDANCE":[]}
    for loc, pos in all_items:
        # 分到最后一个小于 pos 的 header
        target = None
        for hd, start in sorted_hdrs:
            if pos >= start:
                target = hd
        if target:
            segs[target].append(loc)

    # 4. 组装结果
    #  BBOXES: 两两一组
    bbox_locs = segs.get("BBOXES", [])
    bbox_pairs = [(bbox_locs[i], bbox_locs[i+1])
                  for i in range(0, len(bbox_locs)-1, 2)]

    flow_locs = segs.get("FLOW", [])
    aff_locs  = segs.get("AFFORDANCE", [])

    return bbox_pairs, flow_locs, aff_locs

def draw_visual_planning_on_img(img, visual_planning_bins, device, return_only_three_channels=False):
    """
    Draw the visual planning output on the image
    """
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).float().to(device)
    
    bbox_pairs, flow_locs, aff_locs = robust_parse_locs(visual_planning_bins)
    visual_planning_img = np.zeros((3, 224, 224))
    
    for bbox in bbox_pairs:
        top_left, bottom_right = bbox
        bbox_channel = draw_bbox_from_topleft_and_bottomright(top_left, bottom_right)
        visual_planning_img[0] += bbox_channel
    
    # set where > 1 to 1
    visual_planning_img[0] = np.clip(visual_planning_img[0], 0, 1)
    visual_planning_img[1] = draw_flow_from_flow_tokens(flow_locs)
    visual_planning_img[2] = draw_flow_from_flow_tokens(aff_locs)
    
    visual_planning_img = torch.from_numpy(visual_planning_img).float().to(device)
    if return_only_three_channels:
        return visual_planning_img
    else:
        visual_planning_img = torch.cat([img.permute(2,0,1), visual_planning_img], dim=0)  # [6, 224, 224]
        
        return visual_planning_img


def load_radio_model(device, load_from_cache=True, cache_dir="/home/gck/.cache/huggingface/hub"):
    hf_repo = "nvidia/RADIO" # For RADIO-H.

    if load_from_cache:
        print(f"Loading RADIO model from local directory: {cache_dir}")
        image_processor = CLIPImageProcessor.from_pretrained(hf_repo, cache_dir=cache_dir, local_files_only=True)
        model = AutoModel.from_pretrained(hf_repo, cache_dir=cache_dir, trust_remote_code=True, local_files_only=True)
    else:
        print(f"Loading RADIO model from Hugging Face Hub: {hf_repo}")
        image_processor = CLIPImageProcessor.from_pretrained(hf_repo)
        model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
    model.eval().to(device)

    return model, image_processor

def extract_radio_feature(model, image_processor, image, device, patch_size=16):
    assert isinstance(image, Image.Image) or isinstance(image, np.array), "Input image must be a PIL Image or a numpy array."
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(device)

    summary, features = model(pixel_values)
    
    return features

def load_qwen25_tokenizer_and_model(device, load_from_cache=True, cache_dir="/home/gck/.cache/huggingface/hub"):
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    if load_from_cache:
        print(f"Loading Qwen2.5-7B-Instruct model from local directory: {cache_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            output_hidden_states=True,
            torch_dtype="auto",
            local_files_only=True,
            trust_remote_code=True,         # Qwen 需要这个
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
    else:
        print(f"Loading Qwen2.5-7B-Instruct model from Hugging Face Hub: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            torch_dtype="auto",
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def extract_qwen25_language_embedding(model, tokenizer, input_language, device):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": input_language}
    ]
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 直接获取完整输入的 tokens 和 offset mapping
    encoding = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True).to(device)
    input_ids = encoding.input_ids
    offsets = encoding.offset_mapping[0]

    # 根据模板的标记，提取出用户文本所在的字符范围
    start_marker = "<|im_start|>user"
    end_marker = "<|im_end|>"
    start_pos = full_text.find(start_marker)
    end_pos = full_text.find(end_marker, start_pos)
    if start_pos == -1 or end_pos == -1:
        raise ValueError("无法在文本中找到用户输入标记")
    # 用户内容的起始位置（排除标记本身）
    user_text_start = start_pos + len(start_marker)
    user_text = full_text[user_text_start:end_pos].strip()

    # 确定 token 索引范围
    token_start, token_end = None, None
    for i, (s, e) in enumerate(offsets.tolist()):
        # 只对非特殊 token 有效（e.g. s != e)
        if token_start is None and s >= user_text_start:
            token_start = i
        if token_start is not None and e >= end_pos:
            token_end = i + 1  # 包含当前 token
            break

    if token_start is None or token_end is None:
        raise ValueError("未能定位到用户输入对应的 token 范围")
    
    # 获取所有 token 的最后一层 hidden state
    outputs = model(**{k: v for k, v in encoding.items() if k != "offset_mapping"}, output_hidden_states=True)
    last_hidden_state = outputs.hidden_states[-1]
    
    # 只保留用户输入部分的 embedding
    user_embedding = last_hidden_state[:, token_start:token_end, :]
    # print("User embedding shape:", user_embedding.shape)
    
    return user_embedding