MODEL_INFO = {
    'LLaMA3-8B-Chinese-Chat': {
            'model_name_or_path': './models/Meta-Llama-3-8B-Instruct', 
            'finetuning_type': 'lora',
            'quantization_bit': None,
            'quantization_method': 'bitsandbytes', 
            'template': 'llama3', 
            'flash_attn': 'auto',
            'use_unsloth': False,
            'visual_inputs': False, 
            'rope_scaling': None, 
            'infer_backend': 'huggingface', 
            'infer_dtype': 'auto'
    }
}

        t2 = {
            'model_name_or_path': './models/Qwen-7b-sft', 
            'finetuning_type': 'lora', 
            'quantization_bit': None, 
            'quantization_method': 'bitsandbytes', 
            'template': 'qwen', 
            'flash_attn': 'auto', 
            'use_unsloth': False, 
            'visual_inputs': False, 
            'rope_scaling': None, 
            'infer_backend': 'huggingface', 
            'infer_dtype': 'auto', 
            'adapter_name_or_path': 'saves/Qwen-7B-Chat/lora/5000b_wiki_train_202407171518'
        }