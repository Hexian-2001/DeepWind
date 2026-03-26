from transformers import PretrainedConfig
from typing import List
from src.utils.registry import register_config


@register_config("deepwind")
class DeepWindConfig(PretrainedConfig):
    model_type = "deepwind"

    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 2048,
        input_patch_size: int = 16,
        input_patch_stride: int = 16,
        context_length: int = 8192,
        quantiles: List[float] = [
                0.01, 0.05, 0.1, 0.15, 0.2,
                0.25, 0.3, 0.35, 0.4, 0.45,
                0.5,
                0.55, 0.6, 0.65, 0.7, 0.75,
                0.8, 0.85, 0.9, 0.95, 0.99,
            ],
        dropout: float = 0.1,
        # Feature flags
        use_variate_atten = True,
        use_variate_embed = True,
        use_coord_embed = True,
        use_load_balance_loss = True,
        aux_loss_weight = 0.02,
        use_rotary_emb = False,
        use_rms_norm = True,
        variate_atten_every_n_layers = 2, 
        variate_atten_first = False,
        # Magic number management
        num_known_variates: int = 10,  
        num_patch_stats: int = 9,      
        # Backbone specifics
        dense_act_fn: str = "silu",
        use_arcsinh: bool = True,
        use_moe: bool = True,

        num_experts = 4,
        num_experts_per_token = 2,
        **kwargs,
    ):
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.input_patch_size = input_patch_size
        self.input_patch_stride = input_patch_stride
        self.context_length = context_length
        self.quantiles = quantiles
        self.dropout = dropout
        self.use_variate_atten = use_variate_atten
        self.use_variate_embed = use_variate_embed
        self.use_coord_embed = use_coord_embed
        self.use_load_balance_loss = use_load_balance_loss
        self.use_rotary_emb = use_rotary_emb
        self.use_rms_norm = use_rms_norm
        self.variate_atten_every_n_layers = variate_atten_every_n_layers
        self.variate_atten_first = variate_atten_first
        self.num_known_variates = num_known_variates
        self.num_patch_stats = num_patch_stats
        self.dense_act_fn = dense_act_fn
        self.use_arcsinh = use_arcsinh
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.aux_loss_weight = aux_loss_weight
        super().__init__(**kwargs)
        