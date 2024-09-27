from collections import OrderedDict, defaultdict
from enum import Enum
from typing import Dict, Optional

from peft.utils import SAFETENSORS_WEIGHTS_NAME as SAFE_ADAPTER_WEIGHTS_NAME
from peft.utils import WEIGHTS_NAME as ADAPTER_WEIGHTS_NAME
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME


CHECKPOINT_NAMES = {
    SAFE_ADAPTER_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
}

CHOICES = ["A", "B", "C", "D"]

DATA_CONFIG = "dataset_info.json"

DEFAULT_TEMPLATE = defaultdict(str)

FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}

IGNORE_INDEX = -100

LAYERNORM_NAMES = {"norm", "ln"}

LLAMABOARD_CONFIG = "llamaboard_config.yaml"

METHODS = ["full", "freeze", "lora"]

MOD_SUPPORTED_MODELS = {"bloom", "falcon", "gemma", "llama", "mistral", "mixtral", "phi", "starcoder2"}

PEFT_METHODS = {"lora"}

RUNNING_LOG = "running_log.txt"

SUBJECTS = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]

SUPPORTED_MODELS = OrderedDict()

TRAINER_LOG = "trainer_log.jsonl"

TRAINING_ARGS = "training_args.yaml"

TRAINING_STAGES = {
    "Supervised Fine-Tuning": "sft",
    "Reward Modeling": "rm",
    "PPO": "ppo",
    "DPO": "dpo",
    "KTO": "kto",
    "Pre-Training": "pt",
}

STAGES_USE_PAIR_DATA = {"rm", "dpo"}

SUPPORTED_CLASS_FOR_S2ATTN = {"llama"}

V_HEAD_WEIGHTS_NAME = "value_head.bin"

V_HEAD_SAFE_WEIGHTS_NAME = "value_head.safetensors"

VISION_MODELS = set()


class DownloadSource(str, Enum):
    DEFAULT = "hf"
    MODELSCOPE = "ms"


def register_model_group(
    models: Dict[str, Dict[DownloadSource, str]],
    template: Optional[str] = None,
    vision: bool = False,
) -> None:
    prefix = None
    for name, path in models.items():
        if prefix is None:
            prefix = name.split("-")[0]
        else:
            print(prefix, name.split("-")[0])
            assert prefix == name.split("-")[0], "prefix should be identical."
        SUPPORTED_MODELS[name] = path
    if template is not None:
        DEFAULT_TEMPLATE[prefix] = template
    if vision:
        VISION_MODELS.add(prefix)


# api.py 接口支持模型
API_SUPPORTED_MODELS = {
    "LLaMA3-8B-Chat": {
        "id": "LLaMA3-8B-Chat",
        "object": "model",
        "created": 1721810843,
        "owned_by": "owner",
        "description": "Llama 3 系列大型语言模型 (LLM)是一组经过预训练和指令调整的生成文本模型，大小分别为 8B 和 70B。Llama 3 指令调整模型针对对话用例进行了优化，在常见的行业基准上优于许多可用的开源聊天模型。",
        "status": "stopped",
        "model_args": {
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
    },
    "LLaMA3-8B-Chinese-Chat": {
        "id": "LLaMA3-8B-Chinese-Chat",
        "object": "model",
        "created": 1721810843,
        "owned_by": "owner",
        "description": "首个中文微调LLaMa 3模型，基于 Meta-Llama-3-8B-Instruct 模型，使用 ORPO 对其进行了中文微调，从而提高了其在中文问答中的表现。",
        "status": "stopped",
        "model_args": {
            'model_name_or_path': './models/Llama3-8B-Chinese-Chat', 
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
    },
    # 原本是 LLaMA3-8B-Chinese-Chat-lora-sft-gjb5000b
    "GX-8B-Chinese-Chat-gjb5000b": {
        "id": "GX-8B-Chinese-Chat-gjb5000b",
        "object": "model",
        "created": 1721810843,
        "owned_by": "owner",
        "description": "GX-8B-Chinese-Chat-gjb5000b是一个基于GJB5000B标准数据微调的针对GJB5000B标准的指令调整的语言模型，具有各种针对GJB5000B知识问答能力。",
        "status": "stopped",
        "model_args": {
            'model_name_or_path': './models/Llama3-8B-Chinese-Chat', 
            'finetuning_type': 'lora', 
            'quantization_bit': None, 
            'quantization_method': 'bitsandbytes', 
            'template': 'llama3', 
            'flash_attn': 'auto', 
            'use_unsloth': False, 
            'visual_inputs': False, 
            'rope_scaling': None, 
            'infer_backend': 'huggingface', 
            'infer_dtype': 'auto', 
            'adapter_name_or_path': 'saves/LLaMA3-8B-Chinese-Chat/lora/train_2024-07-31-11-32-05'
        }
    },
    
    "Qwen-7B": {
        "id": "Qwen-7B",
        "object": "model",
        "created": 1721810843,
        "owned_by": "owner",
        "description": "通义千问-7B(Qwen-7B) 是阿里云研发的通义千问大模型系列的70亿参数规模的模型。",
        "status": "stopped",
        "model_args": {
            'model_name_or_path': './models/Qwen-7B', 
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
        }
    },
    "GX-7B-Chat-5000B": {
        "id": "GX-7B-Chat-5000B",
        "object": "model",
        "created": 1721810843,
        "owned_by": "owner",
        "description": "GX-7B-Chat-5000B是一个用QWen-7B作为基础模型，用GJB5000B标准数据微调的针对GJB5000B标准的指令调整的语言模型，具有各种针对GJB5000B知识问答能力。",
        "status": "stopped",
        "model_args": {
            'model_name_or_path': './models/Qwen-7B', 
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
            'adapter_name_or_path': 'saves/Qwen-7B/lora/train_2024-07-31-15-46-21-c4-20epoch-pt-sft'
        }
    },
    "GX-8B-Chinese-Chat-zhaobiao": {
        "id": "GX-8B-Chinese-Chat-zhaobiao",
        "object": "model",
        "created": 1721810843,
        "owned_by": "owner",
        "description": "llama3_lora_sft 招投标数据集微调 llama3",
        "status": "stopped",
        "model_args": {
            'model_name_or_path': './models/LLaMA3-8B-Chinese-Chat-lora-sft', 
            'finetuning_type': 'lora', 
            'quantization_bit': None, 
            'quantization_method': 'bitsandbytes', 
            'template': 'default', 
            'flash_attn': 'auto', 
            'use_unsloth': False, 
            'visual_inputs': False, 
            'rope_scaling': None, 
            'infer_backend': 'huggingface', 
            'infer_dtype': 'auto'
        }
    }
}


# register_model_group(
#     models={
#         "Qwen-7B": {
#             DownloadSource.DEFAULT: "Qwen/Qwen-7B",
#             DownloadSource.MODELSCOPE: "qwen/Qwen-7B",
#         },
#         "Qwen-7B-Chat": {
#             DownloadSource.DEFAULT: "Qwen/Qwen-7B-Chat",
#             DownloadSource.MODELSCOPE: "qwen/Qwen-7B-Chat",
#         }
#     },
#     template="qwen"
# )


register_model_group(
    models={
        "Qwen2-7B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-7B-Instruct",
            DownloadSource.MODELSCOPE: "qwen/Qwen2-7B-Instruct",
        },
        "Qwen2-7B": {}
    },
    template="qwen"
)


register_model_group(
    models={
        "LLaMA3-8B": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3-8B",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3-8B",
        },
        "LLaMA3-8B-Chat": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3-8B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3-8B-Instruct",
        },
        "LLaMA3-8B-Chinese-Chat": {
            DownloadSource.DEFAULT: "shenzhi-wang/Llama3-8B-Chinese-Chat",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama3-8B-Chinese-Chat",
        },
        "LLaMA3-Chinese": {

        }
       # "LLaMA3-8B-Chinese-Chat-lora-sft": {
       #
       #  },
       #  "LLaMA3-8B-Chinese-5000B": {
       #
       #  }
    },
    template="llama3",
)

# LLaMA3-8B-Chinese-Chat
# LLaMA3-8B
# Qwen-7B
# Qwen2-7B
# Llama3-Chinese

if __name__ == '__main__':
    print(SUPPORTED_MODELS)
