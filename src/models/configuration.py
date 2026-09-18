from typing import List, Optional
from transformers import PretrainedConfig
from src.utils.registry import register_config

@register_config("deepwind")
class DeepWindConfig(PretrainedConfig):
    model_type = "deepwind"

    def __init__(
        self,
        # --- Transformer Dimensions ---
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        
        # --- Patching & Sequence ---
        input_patch_size: int = 16,
        input_patch_stride: int = 16,
        context_length: int = 8192,
        
        # --- Probabilistic Output ---
        quantiles: List[float] = None,
        
        # --- Feature Switches (Ablations) ---
        use_variate_atten: bool = True,
        use_variate_embed: bool = True,
        use_coord_embed: bool = True,
        use_load_balance_loss: bool = True,
        aux_loss_weight: float = 0.02,
        use_rotary_emb: bool = False,
        use_xpos: bool = False,          
        use_rms_norm: bool = True,
        use_patch_stats: bool = False,    
        
        # --- Attention Logic ---
        variate_atten_every_n_layers: int = 2,
        variate_atten_first: bool = False,
        
        # --- MoE Configuration ---
        use_moe: bool = True,
        num_experts: int = 4,
        num_experts_per_token: int = 2,
        
        # --- Misc & Metadata ---
        num_known_variates: int = 10,
        num_patch_stats: int = 9,
        dense_act_fn: str = "silu",
        use_arcsinh: bool = True,

        # ── Distribution head ──────────────────────────────────────────────
        pred_head_type:    str   = "quantile",   # "quantile" | "student_t" | "zi_beta" | "gmm"
        gmm_components:    int   = 3,            # only used when pred_head_type="gmm"
        channel_loss_weights: Optional[List[float]] = None,  # per-variate loss weight (len V; idx 0 = power); None = equal

        **kwargs,
    ):
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.input_patch_size = input_patch_size
        self.input_patch_stride = input_patch_stride
        self.context_length = context_length
        self.quantiles = quantiles
        
        # Feature Flags
        self.use_variate_atten = use_variate_atten
        self.use_variate_embed = use_variate_embed
        self.use_coord_embed = use_coord_embed
        self.use_load_balance_loss = use_load_balance_loss
        self.aux_loss_weight = aux_loss_weight
        self.use_rotary_emb = use_rotary_emb
        self.use_xpos = use_xpos
        self.use_rms_norm = use_rms_norm
        self.use_patch_stats = use_patch_stats
        
        # Logic & MoE
        self.variate_atten_every_n_layers = variate_atten_every_n_layers
        self.variate_atten_first = variate_atten_first
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        
        # Misc
        self.num_known_variates = num_known_variates
        self.num_patch_stats = num_patch_stats
        self.dense_act_fn = dense_act_fn
        self.use_arcsinh = use_arcsinh

        # Distribution
        self.pred_head_type  = pred_head_type
        self.gmm_components  = gmm_components
        self.channel_loss_weights = channel_loss_weights
        
        super().__init__(**kwargs)