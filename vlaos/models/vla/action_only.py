import torch
import torch.nn as nn
import numpy as np
from vlaos.models.vla import ActionHead
from vlaos.overwatch import initialize_overwatch
from vlaos.models.vla.ode_solver import ODESolver, CFGScaledModel, get_time_discretization
from PIL import Image
from utils.flow_matching_utils import skewed_timestep_sample
from utils.flow_matching.path import CondOTProbPath
from utils.three_d_tokenizer import Encoder

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

class ActionOnlyVLA(nn.Module):
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
        self.training_algo = training_algo
        if self.training_algo == "flow_matching":
            self.path = CondOTProbPath()
            self.skewed_timesteps = skewed_timesteps
        
        self.action_head = ActionHead(
            cfg=cfg, 
            num_layers=self.vlm.llm_backbone.llm.config.num_hidden_layers,
            llm_emb_dim=self.vlm.llm_backbone.llm.model.layers[0].self_attn.k_proj.weight.shape[0],
        )
        
        self.three_d = cfg.three_d
        if self.three_d:
            self.three_d_encoder = Encoder(
                embedding_dim=96,
                nhist=history_len,
                fps_subsampling_factor=cfg.fps_subsampling_factor,
                num_attn_heads=8,
                quaternion_format='xyzw',
            ) # num_head is the layer number to get the gripper feature

    def load_from_checkpoint(self, pretrained_checkpoint):
        model_state_dict = torch.load(pretrained_checkpoint)["model"]
        self.vlm.load_state_dict(model_state_dict["vlm"])
        self.action_head.load_state_dict(model_state_dict["action_head"])
        
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
    
    def action_head_forward(self, batch, vlm_keys, vlm_values, training=True):
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
            t = torch.zeros(batch["actions"].shape[0]).to("cuda")
            predicted_actions = self.action_head(batch["proprios"], torch.zeros_like(batch["actions"]), t, vlm_keys.to("cuda"), vlm_values.to("cuda"), bf16=training, three_d_tokens=three_d_tokens)
            if self.cfg.action_loss_type == "l1":
                action_loss = torch.abs(predicted_actions - batch["actions"]).mean()
            elif self.cfg.action_loss_type == "l2":
                action_loss = torch.pow(predicted_actions - batch["actions"], 2).mean()
            
        elif self.training_algo == "flow_matching":
            if self.skewed_timesteps:
                t = skewed_timestep_sample(batch["actions"].shape[0], device=batch["actions"].device)
            else:
                t = torch.rand(batch["actions"].shape[0]).to(batch["actions"].device)
                                    
            # the action from the dataset is already in [-1, 1], however, the last dim is {0, 1}, TODO: see if we need to change the gripper action
            noise = torch.randn_like(batch["actions"]).to(batch["actions"].device)
            path_sample = self.path.sample(t=t, x_0=noise, x_1=batch["actions"])
            
            x_t = path_sample.x_t
            u_t = path_sample.dx_t                            
            u_t_predicted = self.action_head(batch["proprios"], x_t, t, vlm_keys.to("cuda"), vlm_values.to("cuda"), bf16=training, three_d_tokens=three_d_tokens)
            
            action_loss = torch.pow(u_t_predicted - u_t, 2).mean()
            
            predicted_actions = u_t_predicted
            
        return predicted_actions, action_loss

    def forward(self, batch, training=True, return_vlm_kv=False):
        vlm_keys, vlm_values = self.vlm_forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
        )
            
        predicted_actions, action_loss = self.action_head_forward(batch, vlm_keys, vlm_values, training=training)
        
        planning_losses = {
            "language_planning_loss": 0.,
            "visual_planning_loss": 0.,
            "image_foresight_planning_loss": 0.,
        }
        
        if return_vlm_kv:
            return predicted_actions, planning_losses, action_loss, vlm_keys, vlm_values
        else:
            return predicted_actions, planning_losses, action_loss
        