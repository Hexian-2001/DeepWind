import torch
from peft import LoraConfig, get_peft_model, TaskType

from src.utils.constants import AttentionAxis

def apply_finetune_strategy(model, lora_r=16, lora_alpha=32):
    """
    Applies the 'Decoupled Variate-Expert Adaptation' strategy with Head tuning.
    
    Args:
        model: The pretrained DeepWind model.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        
    Returns:
        peft_model: The model ready for fine-tuning.
    """
    
    # 1. Dynamic Target Selection
    # Derive targets from the instantiated model. This works for every model
    # depth and for non-default attention schedules.
    target_modules_list = []
    variate_layer_indices = [
        i
        for i, layer in enumerate(model.backbone.layers)
        if layer.attention_axis == AttentionAxis.VARIATE
    ]

    if not variate_layer_indices:
        raise ValueError("LoRA strategy requires at least one variate-attention layer.")
    
    for i in variate_layer_indices:
        # Targeting the QKV projection and Output projection in Variate Attention
        target_modules_list.append(f"backbone.layers.{i}.attention.wQKV")
        target_modules_list.append(f"backbone.layers.{i}.attention.wO")
    
    print(f"[Adapter] Targeted LoRA modules ({len(target_modules_list)}): {target_modules_list[0]} ...")

    # 2. Define PEFT Configuration
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, # Adjust if using a specific HF TaskType
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        bias="none",
        
        # [Strategy Part A] LoRA on Variate Attention
        target_modules=target_modules_list,
        
        # [Strategy Part B] Full Fine-tuning on Router AND Head
        # "router" matches: backbone.layers.X.ffn.router
        # "head" matches: head.projector...
        modules_to_save=["router", "head"] 
    )
    
    # 3. Inject Adapters
    peft_model = get_peft_model(model, peft_config)
    
    # 4. Verify Parameter Efficiency
    trainable_params, all_params = peft_model.get_nb_trainable_parameters()
    print(f"[Adapter] Trainable params: {trainable_params:,d} || All params: {all_params:,d} || ratio: {100 * trainable_params / all_params:.2f}%")
    
    return peft_model
