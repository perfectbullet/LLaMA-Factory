import os.path
import subprocess
import time

from typing import TYPE_CHECKING, Dict, Generator, List, Union

from ...extras.constants import PEFT_METHODS
from ...extras.misc import torch_gc
from ...extras.packages import is_gradio_available
from ...ollama_utils import create_ollama_model
from ...train.tuner import export_model
from ..common import GPTQ_BITS, get_save_dir
from ..locales import ALERTS

if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def can_quantize(checkpoint_path: Union[str, List[str]]) -> "gr.Dropdown":
    if isinstance(checkpoint_path, list) and len(checkpoint_path) != 0:
        return gr.Dropdown(value="none", interactive=False)
    else:
        return gr.Dropdown(interactive=True)


def save_model(
    lang: str,
    model_name: str,
    model_path: str,
    finetuning_type: str,
    checkpoint_path: Union[str, List[str]],
    template: str,
    visual_inputs: bool,
    export_size: int,
    export_quantization_bit: int,
    export_quantization_dataset: str,
    export_device: str,
    export_legacy_format: bool,
    export_dir: str,
) -> Generator[str, None, None]:
    error = ""
    if not model_name:
        error = ALERTS["err_no_model"][lang]
    elif not model_path:
        error = ALERTS["err_no_path"][lang]
    elif not export_dir:
        error = ALERTS["err_no_export_dir"][lang]
    elif export_quantization_bit in GPTQ_BITS and not export_quantization_dataset:
        error = ALERTS["err_no_dataset"][lang]
    elif export_quantization_bit not in GPTQ_BITS and not checkpoint_path:
        error = ALERTS["err_no_adapter"][lang]
    elif export_quantization_bit in GPTQ_BITS and isinstance(checkpoint_path, list):
        error = ALERTS["err_gptq_lora"][lang]

    if error:
        gr.Warning(error)
        yield error
        return

    args = dict(
        model_name_or_path=model_path,
        finetuning_type=finetuning_type,
        template=template,
        visual_inputs=visual_inputs,
        export_dir=export_dir,
        export_hub_model_id=None,
        export_size=export_size,
        export_quantization_bit=int(export_quantization_bit) if export_quantization_bit in GPTQ_BITS else None,
        export_quantization_dataset=export_quantization_dataset,
        export_device=export_device,
        export_legacy_format=export_legacy_format,
    )

    if checkpoint_path:
        if finetuning_type in PEFT_METHODS:  # list
            args["adapter_name_or_path"] = ",".join(
                [get_save_dir(model_name, finetuning_type, adapter) for adapter in checkpoint_path]
            )
        else:  # str
            args["model_name_or_path"] = get_save_dir(model_name, finetuning_type, checkpoint_path)

    yield ALERTS["info_exporting"][lang]
    export_model(args)
    torch_gc()
    yield ALERTS["info_exported"][lang]


def publish_model(
    model_name: str,
    template: str,
    export_device: str,
    export_dir: str,
):
    '''
    publish model to ollama
    '''
    error = ""
    lang = 'zh'
    print('args is {},{},{},{}'.format(model_name, template, export_device, export_dir))
    if not model_name:
        error = ALERTS["err_no_model"][lang]

    elif not export_dir:
        error = ALERTS["err_no_export_dir"][lang]
    if error:
        gr.Warning(error)
        return error
    create_ollama_cmd_msg = create_ollama_model(export_dir)
    return create_ollama_cmd_msg


def create_export_tab(engine: "Engine") -> Dict[str, "Component"]:
    with gr.Row():
        export_size = gr.Slider(minimum=1, maximum=10, value=4, step=1)
        export_quantization_bit = gr.Dropdown(choices=["none"] + GPTQ_BITS, value="none")
        export_quantization_dataset = gr.Textbox(value="data/c4_demo.json")
        export_device = gr.Radio(
            choices=["cpu", "auto"],
            value="auto",
            label="导出设备",
            info="导出模型使用的设备类型"
        )
        export_legacy_format = gr.Checkbox()

    with gr.Row():
        export_dir = gr.Textbox()

    checkpoint_path: gr.Dropdown = engine.manager.get_elem_by_id("top.checkpoint_path")
    checkpoint_path.change(can_quantize, [checkpoint_path], [export_quantization_bit], queue=False)

    export_btn = gr.Button()
    publish_btn = gr.Button('发布模型')

    info_box = gr.Textbox(show_label=False, interactive=False)

    export_btn.click(
        save_model,
        [
            engine.manager.get_elem_by_id("top.lang"),
            engine.manager.get_elem_by_id("top.model_name"),
            engine.manager.get_elem_by_id("top.model_path"),
            engine.manager.get_elem_by_id("top.finetuning_type"),
            engine.manager.get_elem_by_id("top.checkpoint_path"),
            engine.manager.get_elem_by_id("top.template"),
            engine.manager.get_elem_by_id("top.visual_inputs"),
            export_size,
            export_quantization_bit,
            export_quantization_dataset,
            export_device,
            export_legacy_format,
            export_dir,
        ],
        [info_box],
    )

    publish_btn.click(
        publish_model,
        inputs=[
            engine.manager.get_elem_by_id("top.model_name"),
            engine.manager.get_elem_by_id("top.template"),
            export_device,
            export_dir,
        ],
        outputs=[info_box, ],
    )

    return dict(
        export_size=export_size,
        export_quantization_bit=export_quantization_bit,
        export_quantization_dataset=export_quantization_dataset,
        export_device=export_device,
        export_legacy_format=export_legacy_format,
        export_dir=export_dir,
        export_btn=export_btn,
        info_box=info_box,
    )
