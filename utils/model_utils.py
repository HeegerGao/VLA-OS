"""
load.py

Entry point for loading pretrained VLMs for inference; exposes functions for listing available models (with canonical
IDs, mappings to paper experiments, and short descriptions), as well as for loading models (from disk or HF Hub).
"""

import torch
from vlaos.overwatch import initialize_overwatch
from vlaos.models import Qwen25LLMBackbone, DinoSigLIPViTBackbone, QwenVLM
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

def load_qwen_vlm(
    model_id,
    vision_backbone_id,
    image_resize_strategy,
    default_image_size,
    image_sequence_len,
    llm_backbone_id,
    inference_mode,
    stage,
    pretrained_checkpoint,
    run_dir,
    use_flash_attention_2=True,
    load_pretrained_weights_from_hf=True,
):
    # --------------------------------------- Model -----------------------------------------------------
            
    # Load Vision Backbone --> on CPU, in Full Precision (initializing model, image_transform via TIMM)
    overwatch.info(f"Loading Vision Backbone [bold]{vision_backbone_id}[/] via TIMM ")
    vision_backbone = DinoSigLIPViTBackbone(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size, image_sequence_len=image_sequence_len, pretrained=load_pretrained_weights_from_hf)
    image_transform = vision_backbone.get_image_transform()
    
    # Load LLM Backbone --> on CPU, in Full Precision (initializing Tokenizer + handling special tokens if necessary)
    overwatch.info(f"Loading Pretrained LLM [bold]{llm_backbone_id}[/] via HF Transformers")
    llm_backbone = Qwen25LLMBackbone(
        llm_backbone_id,
        inference_mode=inference_mode,
        use_flash_attention_2=use_flash_attention_2,
        pretrained=load_pretrained_weights_from_hf,
    )
    tokenizer = llm_backbone.get_tokenizer()
    
    # Create VLM => wraps `vision_backbone` and `llm`
    overwatch.info(f"Instantiating VLM `{model_id}` for Training Stage = `{stage}`")
    vlm = QwenVLM(
        model_id,
        vision_backbone,
        llm_backbone,
    )

    # Load Weights from Checkpoint (depends on stage, config)
    if pretrained_checkpoint is not None:
        vlm.load_from_checkpoint(stage, run_dir, pretrained_checkpoint=pretrained_checkpoint)
    
    return vlm, image_transform, tokenizer

def get_qwen_planning_input_ids_for_training(
    language_instruction,
    prompt_builder,
    tokenizer,
    device="cuda:0",
    perform_langauge_planning=False,
    perform_visual_planning=False,
    visual_planning_tokenizer=None
):  
    # a batch of inputs
    language_planning_input_ids_batch = []
    visual_planning_input_ids_batch = []
    for lang in language_instruction:
        # clear the prompt builder
        prompt_builder.prompt = ""
        prompt_builder.turn_count = 0
        conversation = []
        conversation.extend(
            [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
            ]
        )
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        if perform_langauge_planning:
            language_planning_input_ids = tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
            language_planning_input_ids = torch.tensor(language_planning_input_ids).to(device)
            language_planning_input_ids_batch.append(language_planning_input_ids)

        if perform_visual_planning:
            visual_planning_input_ids = visual_planning_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
            visual_planning_input_ids = torch.tensor(visual_planning_input_ids).to(device)
            visual_planning_input_ids_batch.append(visual_planning_input_ids)

    # pad sequence
    if perform_langauge_planning:
        language_planning_input_ids = pad_sequence(language_planning_input_ids_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
        language_planning_input_ids = language_planning_input_ids.to(device)
        language_planning_attention_mask = language_planning_input_ids.ne(tokenizer.pad_token_id).long()
        language_planning_attention_mask = language_planning_attention_mask.to(device)
    else:
        language_planning_input_ids, language_planning_attention_mask = None, None
        
    if perform_visual_planning:
        visual_planning_input_ids = pad_sequence(visual_planning_input_ids_batch, batch_first=True, padding_value=visual_planning_tokenizer.pad_token_id)
        visual_planning_input_ids = visual_planning_input_ids.to(device)
        visual_planning_attention_mask = visual_planning_input_ids.ne(visual_planning_tokenizer.pad_token_id).long()
        visual_planning_attention_mask = visual_planning_attention_mask.to(device)
    else:
        visual_planning_input_ids, visual_planning_attention_mask = None, None

    return language_planning_input_ids, language_planning_attention_mask, visual_planning_input_ids, visual_planning_attention_mask