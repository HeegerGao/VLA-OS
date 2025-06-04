"""
load.py

Entry point for loading pretrained VLMs for inference; exposes functions for listing available models (with canonical
IDs, mappings to paper experiments, and short descriptions), as well as for loading models (from disk or HF Hub).
"""

from vlaos.overwatch import initialize_overwatch
from vlaos.models import Qwen25LLMBackbone, DinoSigLIPViTBackbone, QwenVLM
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

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
