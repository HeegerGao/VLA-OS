import torch
import torch.nn as nn
import numpy as np
from vlaos.models.vla import LowLevelActionHead
from vlaos.overwatch import initialize_overwatch
from vlaos.models.vla.ode_solver import ODESolver, CFGScaledModel, get_time_discretization
from vlaos.models.vla.language_planning_head import LanguagePlanningHead
from vlaos.models.vla.visual_planning_head import VisualPlanningHead
from vlaos.models.vla.image_foresight_head import ImageForesightPlanningHead
from PIL import Image
from utils.flow_matching_utils import skewed_timestep_sample
from utils.flow_matching.path import CondOTProbPath
from utils.model_utils import get_qwen_planning_input_ids_for_training
from copy import deepcopy
from utils.utils import draw_visual_planning_on_img
from utils.utils import load_qwen25_tokenizer_and_model, extract_qwen25_language_embedding
from vlaos.models import DinoSigLIPViTBackbone
from vlaos.models.vla.nn_utils import pad_and_make_mask
from utils.three_d_tokenizer import Encoder

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class HierarchicalVLA(nn.Module):
    ''' The VLM will output the planning results first, then the action head will use these results to predict the actions
        There is no gradient between the VLM and the action head.
        The action head also uses the extracted embeddings from the VLM.
    '''
    def __init__(
        self,
        cfg,
        vlm,
        training_algo: str = "bc",
        skewed_timesteps: bool = False,
        history_len: int = 2,
    ) -> None:
        super().__init__()
        
        self.cfg = cfg
        self.vlm = vlm
        self.vlm.freeze_backbone("freeze")
        self.training_algo = training_algo
        if self.training_algo == "flow_matching":
            self.path = CondOTProbPath()
            self.skewed_timesteps = skewed_timesteps
        
        self.action_head = LowLevelActionHead(
            cfg=cfg, 
            num_layers=self.vlm.llm_backbone.llm.config.num_hidden_layers,
            language_planning="language_planning" in self.cfg.planning_heads,
            visual_planning="visual_planning" in self.cfg.planning_heads,
            image_foresight_planning="image_foresight_planning" in self.cfg.planning_heads,
        )
        self.action_head.qw25_model, self.action_head.qw25_tokenizer = load_qwen25_tokenizer_and_model("cpu")
        self.action_head.qw25_model.eval()
        self.action_head.qw25_model.requires_grad_(False)
        
        self.three_d = cfg.three_d
        if self.three_d:
            self.three_d_encoder = Encoder(
                embedding_dim=96,
                nhist=history_len,
                fps_subsampling_factor=cfg.fps_subsampling_factor,
                num_attn_heads=8,
                quaternion_format='xyzw',
            ) # num_head is the layer number to get the gripper feature
        
        
        self.action_head.radio_model = DinoSigLIPViTBackbone(
            "dinosiglip-vit-so-224px",
            "resize-naive",
            default_image_size=224, 
            image_sequence_len=cfg.image_sequence_len,  # must be given by the cfg
            pretrained=False
        )
        self.action_head.radio_processor = self.action_head.radio_model.get_image_transform()
        
        # planning mode is implicit
        self.planning_mode = self.cfg.planning_mode
        # because if we set it to explicit, the planning head will have additional network modules and parameters
        assert self.planning_mode == "implicit", "Only implicit planning mode is supported for Hierarchcial VLA."
        
        if "language_planning" in self.cfg.planning_heads:
            self.language_planning_head = LanguagePlanningHead(
                vocab_size=self.vlm.llm_backbone.llm.config.vocab_size,
                hidden_size=self.cfg.language_planning_head.hidden_size,
                num_layers=self.vlm.llm_backbone.llm.config.num_hidden_layers,
                llm_emb_dim=self.vlm.llm_backbone.llm.model.layers[0].self_attn.k_proj.weight.shape[0],
                num_heads=self.cfg.language_planning_head.num_heads,
                planning_mode=self.planning_mode,
            )
            
        if "visual_planning" in self.cfg.planning_heads:
            self.visual_planning_head = VisualPlanningHead(
                vocab_size=self.vlm.llm_backbone.llm.config.vocab_size,
                visual_planning_vocab_size=self.cfg.visual_planning_head.visual_planning_vocab_size,
                hidden_size=self.cfg.visual_planning_head.hidden_size,
                num_layers=self.vlm.llm_backbone.llm.config.num_hidden_layers,
                llm_emb_dim=self.vlm.llm_backbone.llm.model.layers[0].self_attn.k_proj.weight.shape[0],
                num_heads=self.cfg.visual_planning_head.num_heads,
                planning_mode=self.planning_mode,
            )
            self.visual_planning_tokenizer = deepcopy(self.vlm.llm_backbone.tokenizer)
            self.visual_planning_tokenizer.add_tokens([f"<loc_{i}>" for i in range(self.cfg.visual_planning_head.visual_planning_vocab_size)])
        
        if "image_foresight_planning" in self.cfg.planning_heads:
            self.image_foresight_head = ImageForesightPlanningHead(
                vae_cfg=self.cfg.image_foresight_planning_head.vae_cfg,
                generation_head_cfg=self.cfg.image_foresight_planning_head.generation_head_cfg,
                bsc_cfg=self.cfg.image_foresight_planning_head.bsc_cfg,
            )
            
    def load_from_checkpoint(self, high_level_checkpoint=None, low_level_checkpoint=None):
        if high_level_checkpoint is not None:
            high_level_model_state_dict = torch.load(high_level_checkpoint)["model"]
            self.vlm.load_state_dict(high_level_model_state_dict["vlm"])
            if "language_planning" in self.cfg.planning_heads:
                self.language_planning_head.load_state_dict(high_level_model_state_dict["language_planning_head"])
                print("Language Planning Head Loaded")
            if "visual_planning" in self.cfg.planning_heads:
                self.visual_planning_head.load_state_dict(high_level_model_state_dict["visual_planning_head"])
                print("Visual Planning Head Loaded")
            if "image_foresight_planning" in self.cfg.planning_heads:
                self.image_foresight_head.load_state_dict(high_level_model_state_dict["image_foresight_head"], strict=False)
                print("Image Foresight Planning Head Loaded")
        
        if low_level_checkpoint is not None:
            low_level_model_state_dict = torch.load(low_level_checkpoint)["model"]
            self.action_head.load_state_dict(low_level_model_state_dict["action_head"], strict=False)

        return

    def get_kv_cache_from_vlm(self, vlm_output):
        vlm_num_layers = len(vlm_output["past_key_values"])
        vlm_keys, vlm_values = [], []
        for layer_num in range(vlm_num_layers):
            vlm_keys.append(vlm_output["past_key_values"][layer_num][0])    # [bsz, num_kv_heads, seq_len, kv_head_dim]
            vlm_values.append(vlm_output["past_key_values"][layer_num][1])  # [bsz, num_kv_heads, seq_len, kv_head_dim]
        
        vlm_keys = torch.stack(vlm_keys, dim=0)  # [num_layers, bsz, num_kv_heads, seq_len, kv_head_dim]
        vlm_values = torch.stack(vlm_values, dim=0)  # [num_layers, bsz, num_kv_heads, seq_len, kv_head_dim]
        
        vlm_keys = vlm_keys.permute(0, 1, 3, 2, 4)  # [num_layers, bsz, seq_len, num_kv_heads, kv_head_dim]
        vlm_values = vlm_values.permute(0, 1, 3, 2, 4)  # [num_layers, bsz, seq_len, num_kv_heads, kv_head_dim]
        
        vlm_keys = vlm_keys.reshape(vlm_num_layers, vlm_keys.size(1), vlm_keys.size(2), -1)  # [num_layers, bsz, seq_len, num_kv_heads*kv_head_dim]
        vlm_values = vlm_values.reshape(vlm_num_layers, vlm_values.size(1), vlm_values.size(2), -1)  # [num_layers, bsz, seq_len, num_kv_heads*kv_head_dim]

        return vlm_keys, vlm_values

    def vlm_forward(self, input_ids, attention_mask, pixel_values, labels):
        # [Contract] self.vlm.forward() must automatically compute `loss` and return!       
        vlm_output = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            use_cache=True, # let it return past_key_values
        )

        vlm_keys, vlm_values = self.get_kv_cache_from_vlm(vlm_output)
        
        return vlm_keys, vlm_values

    def forward(
        self,
        batch,
    ):
        planning_losses = {
            "language_planning_loss": 0.,
            "visual_planning_loss": 0.,
            "image_foresight_planning_loss": 0.,
        }
                
        device = batch["input_ids"].device
        batch_size = batch["input_ids"].shape[0]
        with torch.set_grad_enabled(False):
            vlm_keys, vlm_values = self.vlm_forward(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
            )
            
            # prepare language planning or visual planning input ids
            if "language_planning" in self.cfg.planning_heads or "visual_planning" in self.cfg.planning_heads or "image_foresight_planning" in self.cfg.planning_heads:
                # prepare planning input ids
                language_planning_input_ids, language_planning_attention_mask, visual_planning_input_ids, visual_planning_attention_mask = get_qwen_planning_input_ids_for_training(
                    batch["language_description"],
                    prompt_builder=self.vlm.llm_backbone.prompt_builder_fn("qwen"),
                    tokenizer=self.vlm.llm_backbone.tokenizer,
                    device=device,
                    perform_langauge_planning="language_planning" in self.cfg.planning_heads,
                    perform_visual_planning="visual_planning" in self.cfg.planning_heads,
                    visual_planning_tokenizer=self.visual_planning_tokenizer if "visual_planning" in self.cfg.planning_heads else None,
                )
                
            # planning head inference
            if "language_planning" in self.cfg.planning_heads:
                generated_ids, _, _ = self.language_planning_head.generate(
                    input_ids=language_planning_input_ids,
                    attention_mask=language_planning_attention_mask,
                    vlm_keys=vlm_keys,
                    vlm_values=vlm_values,
                    eos_token_id=self.vlm.llm_backbone.tokenizer.eos_token_id,
                )
                input_len = language_planning_input_ids.shape[1]
                new_tokens = generated_ids[:, input_len:]
                decoded = self.vlm.llm_backbone.tokenizer.batch_decode(new_tokens.tolist(), skip_special_tokens=True)
                batch["language_planning_reasoning"] = [t.strip() for t in decoded]

                language_planning = []
                for lang in batch["language_planning_reasoning"]:
                    language_planning.append(
                        extract_qwen25_language_embedding(
                            self.action_head.qw25_model,
                            self.action_head.qw25_tokenizer,
                            lang,
                            device=device,
                        )[0]
                    )
                language_planning_embedding, language_planning_feature_attention_mask = pad_and_make_mask(language_planning)
                language_planning_embedding = language_planning_embedding.to(device)
                language_planning_feature_attention_mask = language_planning_feature_attention_mask.to(device)
            else:
                language_planning_embedding = None
                language_planning_feature_attention_mask = None
                
            if "visual_planning" in self.cfg.planning_heads:
                batch_theee_channels = []
                generated_bin_ids, visual_planning_keys, visual_planning_values = self.visual_planning_head.generate(
                    input_ids=visual_planning_input_ids,
                    attention_mask=visual_planning_attention_mask,
                    vlm_keys=vlm_keys,
                    vlm_values=vlm_values,
                    eos_token_id=self.visual_planning_tokenizer.eos_token_id,
                )
                input_len = visual_planning_input_ids.shape[1]
                new_tokens = generated_bin_ids[:, input_len:]
                generated_bins = self.visual_planning_tokenizer.batch_decode(new_tokens.tolist(), skip_special_tokens=True)
                for i in range(batch_size):
                    batch_theee_channels.append(draw_visual_planning_on_img(None, generated_bins[i].strip(), device=device, return_only_three_channels=True))

                visual_planning_imgs = []
                view_num = batch["original_imgs"].shape[1] // batch["proprios"].shape[1]
                current_primary_original_imgs = batch["original_imgs"][:, -view_num, :, :, :].clone()    # this means extracting tensors with the last `view_num`-th index, i.e., the last primary view image
                for i in range(batch_size):
                    three_channels = batch_theee_channels[i]
                    visual_planning_imgs.append(torch.cat([current_primary_original_imgs[i], three_channels], dim=0))   # [6, 224, 224]
                visual_planning_imgs = torch.stack(visual_planning_imgs).to(device)    # [b, 6, 224, 224]
            else:
                visual_planning_imgs = None
                
            if "image_foresight_planning" in self.cfg.planning_heads:
                self.image_foresight_head.image_generation_head.use_flex_attn = False

                generated_image_list, _, _ = self.image_foresight_head.generate(
                    vlm_keys,
                    vlm_values,
                    B=batch_size,
                    cfg_list=1,
                    tau_list=0.5,
                    scale_schedule=[
                        [1, 1, 1],
                        [1, 2, 2],
                        [1, 4, 4],
                        [1, 6, 6],
                        [1, 8, 8],
                        [1, 12, 12],
                        [1, 16, 16]
                    ],
                    cfg_insertion_layer=[0],
                    vae_type=32,
                    g_seed=42,
                )
                
                mean = torch.tensor([0.4850, 0.4560, 0.4060]).to(device)
                std = torch.tensor([0.2290, 0.2240, 0.2250]).to(device)

                goal_image = (generated_image_list.permute(0, 3, 1, 2).float().div(255/2).sub(1) - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
            else:
                goal_image = None

        # action head forward
        if self.three_d:
            three_d_tokens = self.three_d_encoder(
                eef_pose_history = batch["proprios"][:, :, :8], # first 7 dims are eef pose
                multi_RGB_dict = batch["rgb_dict"],
                multi_depth_dict = batch["depth_dict"],
                extrinsics_dict = batch["extrinsics_dict"],
                intrinsics_dict = batch["intrinsics_dict"],
            )
        else:
            three_d_tokens = None        
        
        if self.training_algo == "bc":
            t = torch.zeros(batch["actions"].shape[0]).to(device)
            predicted_actions = self.action_head(
                vlm_keys,
                vlm_values,
                batch["proprios"],
                torch.zeros_like(batch["actions"]),
                t, 
                language_planning_features=language_planning_embedding,
                visual_planning_imgs=visual_planning_imgs,
                image_foresight_img=goal_image,
                language_planning_feature_attention_mask=language_planning_feature_attention_mask,
                bf16=True,
                three_d_tokens=three_d_tokens,
            )
            if self.cfg.action_loss_type == "l1":
                action_loss = torch.abs(predicted_actions - batch["actions"]).mean()
            elif self.cfg.action_loss_type == "l2":
                action_loss = torch.pow(predicted_actions - batch["actions"], 2).mean()
                
        elif self.training_algo == "flow_matching":
            if self.skewed_timesteps:
                t = skewed_timestep_sample(batch_size, device=device)
            else:
                t = torch.rand(batch_size).to(device)
                                    
            # the action from the dataset is already in [-1, 1], however, the last dim is {0, 1}, TODO: see if we need to change the gripper action
            noise = torch.randn_like(batch["actions"]).to(device)
            path_sample = self.path.sample(t=t, x_0=noise, x_1=batch["actions"])
            
            x_t = path_sample.x_t
            u_t = path_sample.dx_t                            
            u_t_predicted = self.action_head(
                vlm_keys,
                vlm_values,
                batch["proprios"],
                x_t,
                t, 
                language_planning_features=language_planning_embedding,
                visual_planning_imgs=visual_planning_imgs,
                image_foresight_img=goal_image,
                language_planning_feature_attention_mask=language_planning_feature_attention_mask,
                bf16=True,
                three_d_tokens=three_d_tokens,
            )
            
            if self.cfg.action_loss_type == "l1":
                action_loss = torch.abs(u_t_predicted - u_t).mean()
            elif self.cfg.action_loss_type == "l2":
                action_loss = torch.pow(u_t_predicted - u_t, 2).mean()
        
        return predicted_actions, planning_losses, action_loss
