import torch
from vlaos.overwatch import initialize_overwatch
from vlaos.models.vla.action_only import ActionOnlyVLA
from vlaos.models.vla.language_planning_head import LanguagePlanningHead
from vlaos.models.vla.visual_planning_head import VisualPlanningHead
from vlaos.models.vla.image_foresight_head import ImageForesightPlanningHead
from utils.utils import draw_visual_planning_on_img
from PIL import Image
from copy import deepcopy

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class IntegratedVLA(ActionOnlyVLA):
    ''' Same architecture with the ActionOnlyVLA but use extra planning heads for language or visual planning
    '''
    def __init__(
        self,
        cfg,
        vlm,
        training_algo: str = "bc",
        skewed_timesteps: bool = False,
        history_len: int = 2,
    ) -> None:
        super().__init__(cfg, vlm, training_algo, skewed_timesteps, history_len)
        
        assert len(self.cfg.planning_heads) > 0, "At least one planning head should be specified"
        
        self.planning_mode = self.cfg.planning_mode
        
        if "language_planning" in self.cfg.planning_heads:
            self.language_planning_head = LanguagePlanningHead(
                vocab_size=self.vlm.llm_backbone.llm.config.vocab_size,
                hidden_size=self.cfg.language_planning_head.hidden_size,
                num_layers=self.vlm.llm_backbone.llm.config.num_hidden_layers,
                llm_emb_dim=self.vlm.llm_backbone.llm.model.layers[0].self_attn.k_proj.weight.shape[0],
                num_heads=self.cfg.language_planning_head.num_heads,
                planning_mode=self.cfg.planning_mode,
            )
        if "visual_planning" in self.cfg.planning_heads:
            self.visual_planning_head = VisualPlanningHead(
                vocab_size=self.vlm.llm_backbone.llm.config.vocab_size,
                visual_planning_vocab_size=self.cfg.visual_planning_head.visual_planning_vocab_size,
                hidden_size=self.cfg.visual_planning_head.hidden_size,
                num_layers=self.vlm.llm_backbone.llm.config.num_hidden_layers,
                llm_emb_dim=self.vlm.llm_backbone.llm.model.layers[0].self_attn.k_proj.weight.shape[0],
                num_heads=self.cfg.visual_planning_head.num_heads,
                planning_mode=self.cfg.planning_mode,
            )
            self.visual_planning_tokenizer = deepcopy(self.vlm.llm_backbone.tokenizer)
            self.visual_planning_tokenizer.add_tokens([f"<loc_{i}>" for i in range(self.cfg.visual_planning_head.visual_planning_vocab_size)])
            
        if "image_foresight_planning" in self.cfg.planning_heads:
            self.image_foresight_head = ImageForesightPlanningHead(
                vae_cfg=self.cfg.image_foresight_planning_head.vae_cfg,
                generation_head_cfg=self.cfg.image_foresight_planning_head.generation_head_cfg,
                bsc_cfg=self.cfg.image_foresight_planning_head.bsc_cfg,
            )
            self.image_foresight_head.image_generation_head.special_init(
                aln_init=1e-3,
                aln_gamma_init=5e-6,
                scale_head=0.02,
                scale_proj=1,
            )

    def load_from_checkpoint(self, pretrained_checkpoint):
        model_state_dict = torch.load(pretrained_checkpoint)["model"]
        self.vlm.load_state_dict(model_state_dict["vlm"])
        self.action_head.load_state_dict(model_state_dict["action_head"])
        if "language_planning" in self.cfg.planning_heads:
            self.language_planning_head.load_state_dict(model_state_dict["language_planning_head"])
            print("Language Planning Head Loaded")
        if "visual_planning" in self.cfg.planning_heads:
            self.visual_planning_head.load_state_dict(model_state_dict["visual_planning_head"])
            print("Visual Planning Head Loaded")
        if "image_foresight_planning" in self.cfg.planning_heads:
            self.image_foresight_head.load_state_dict(model_state_dict["image_foresight_head"])
            print("Image Foresight Planning Head Loaded")

        return

    def forward(self, batch, training=True):
        planning_losses = {
            "language_planning_loss": 0.,
            "visual_planning_loss": 0.,
            "image_foresight_planning_loss": 0.,
        }
        
        vlm_keys, vlm_values = self.vlm_forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
        )
        
        # planning head forward
        if "language_planning" in self.cfg.planning_heads:
            language_planning_output, language_planning_keys, language_planning_values = self.language_planning_head(
                vlm_key=vlm_keys, 
                vlm_value=vlm_values,
                input_ids=batch["language_planning_input_ids"],
                attention_mask=batch["language_planning_attention_mask"],
                labels=batch["language_planning_labels"],
            )
            language_planning_loss = language_planning_output.loss
            planning_losses["language_planning_loss"] = language_planning_loss
            
        if "visual_planning" in self.cfg.planning_heads:
            visual_planning_output, visual_planning_keys, visual_planning_values = self.visual_planning_head(
                vlm_key=vlm_keys, 
                vlm_value=vlm_values,
                input_ids=batch["visual_planning_input_ids"],
                attention_mask=batch["visual_planning_attention_mask"],
                labels=batch["visual_planning_labels"],
            )
            visual_planning_loss = visual_planning_output.loss
            planning_losses["visual_planning_loss"] = visual_planning_loss
            
        if "image_foresight_planning" in self.cfg.planning_heads:
            image_foresight_loss, if_keys, if_values = self.image_foresight_head(
                vlm_key=vlm_keys,
                vlm_value=vlm_values,
                future_images=batch["goal_image"],
            )
            planning_losses["image_foresight_planning_loss"] = image_foresight_loss

        if self.planning_mode == "implicit":
            predicted_actions, action_loss = self.action_head_forward(batch, vlm_keys, vlm_values, training)
            
        else:
            if "language_planning" in self.cfg.planning_heads:
                vlm_keys = torch.cat([vlm_keys, language_planning_keys], dim=2)
                vlm_values = torch.cat([vlm_values, language_planning_values], dim=2)
            if "visual_planning" in self.cfg.planning_heads:
                vlm_keys = torch.cat([vlm_keys, visual_planning_keys], dim=2)
                vlm_values = torch.cat([vlm_values, visual_planning_values], dim=2)
            if "image_foresight_planning" in self.cfg.planning_heads:
                vlm_keys = torch.cat([vlm_keys, if_keys], dim=2)
                vlm_values = torch.cat([vlm_values, if_values], dim=2)

            predicted_actions, action_loss = self.action_head_forward(batch, vlm_keys, vlm_values, training)
        
        return predicted_actions, planning_losses, action_loss 
