import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from loguru import logger

from ...data.parse_file import read_excel, read_json, save2json
from ...extras.constants import DATA_CONFIG
from ...extras.packages import is_gradio_available

if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component

PAGE_SIZE = 2


def prev_page(page_index: int) -> int:
    return page_index - 1 if page_index > 0 else page_index


def next_page(page_index: int, total_num: int) -> int:
    return page_index + 1 if (page_index + 1) * PAGE_SIZE < total_num else page_index


def can_preview(dataset_dir: str, dataset: list) -> "gr.Button":
    try:
        with open(os.path.join(dataset_dir, DATA_CONFIG), "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    except Exception:
        return gr.Button(interactive=False)

    if len(dataset) == 0 or "file_name" not in dataset_info[dataset[0]]:
        return gr.Button(interactive=False)

    data_path = os.path.join(dataset_dir, dataset_info[dataset[0]]["file_name"])
    if os.path.isfile(data_path) or (os.path.isdir(data_path) and os.listdir(data_path)):
        return gr.Button(interactive=True)
    else:
        return gr.Button(interactive=False)


def _load_data_file(file_path: str) -> List[Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith(".json"):
            return json.load(f)
        elif file_path.endswith(".jsonl"):
            return [json.loads(line) for line in f]
        else:
            return list(f)


def get_preview(dataset_dir: str, dataset: list, page_index: int) -> Tuple[int, list, "gr.Column"]:
    with open(os.path.join(dataset_dir, DATA_CONFIG), "r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    data_path = os.path.join(dataset_dir, dataset_info[dataset[0]]["file_name"])
    if os.path.isfile(data_path):
        data = _load_data_file(data_path)
    else:
        data = []
        for file_name in os.listdir(data_path):
            data.extend(_load_data_file(os.path.join(data_path, file_name)))

    return len(data), data[PAGE_SIZE * page_index: PAGE_SIZE * (page_index + 1)], gr.Column(visible=True)


def create_preview_box(dataset_dir: "gr.Textbox", dataset: "gr.Dropdown") -> Dict[str, "Component"]:
    data_preview_btn = gr.Button(interactive=False, scale=1)
    with gr.Column(visible=False, elem_classes="modal-box") as preview_box:
        with gr.Row():
            preview_count = gr.Number(value=0, interactive=False, precision=0)
            page_index = gr.Number(value=0, interactive=False, precision=0)

        with gr.Row():
            prev_btn = gr.Button()
            next_btn = gr.Button()
            close_btn = gr.Button()

        with gr.Row():
            preview_samples = gr.JSON()

    dataset.change(can_preview, [dataset_dir, dataset], [data_preview_btn], queue=False).then(
        lambda: 0, outputs=[page_index], queue=False
    )
    data_preview_btn.click(
        get_preview, [dataset_dir, dataset, page_index], [preview_count, preview_samples, preview_box], queue=False
    )
    prev_btn.click(prev_page, [page_index], [page_index], queue=False).then(
        get_preview, [dataset_dir, dataset, page_index], [preview_count, preview_samples, preview_box], queue=False
    )
    next_btn.click(next_page, [page_index, preview_count], [page_index], queue=False).then(
        get_preview, [dataset_dir, dataset, page_index], [preview_count, preview_samples, preview_box], queue=False
    )
    close_btn.click(lambda: gr.Column(visible=False), outputs=[preview_box], queue=False)
    return dict(
        data_preview_btn=data_preview_btn,
        preview_count=preview_count,
        page_index=page_index,
        prev_btn=prev_btn,
        next_btn=next_btn,
        close_btn=close_btn,
        preview_samples=preview_samples,
    )


def import_data(dataset_dir: str, dataset: list, inputs: gr.components.File):
    xlsx_path = '{}'.format(inputs)
    fname = os.path.basename(xlsx_path)
    if not fname.endswith('.xlsx'):
        gr.Warning('当前必须使用 xlsx格式文件')
        return
    new_json_path = os.path.join(dataset_dir, fname.replace('.xlsx', '.json'))
    if os.path.exists(new_json_path):
        print('{} is existed'.format(new_json_path))
        gr.Warning('文件已存在: {}'.format(new_json_path))
        return
    new_json_path, dataset_name = read_excel(new_json_path, xlsx_path)
    if dataset_name == '':
        gr.Warning('数据格式错误')
        return
    gr.Info('json {}'.format(new_json_path))

    data_info_name = 'dataset_info.json'
    data_info_path = os.path.join(dataset_dir, data_info_name)
    data_info_json = read_json(data_info_path)
    logger.info('data_info_json is {}', len(data_info_json))

    if dataset_name.startswith('c4'):
        c4_format = {
            "file_name": os.path.basename(new_json_path),
            "columns": {
              "prompt": "text"
            }
        }
        # update data_info and save
        data_info_json[dataset_name] = c4_format
    else:
        alpaca_format = {
            "file_name": os.path.basename(new_json_path)
        }
        # update data_info and save
        data_info_json[dataset_name] = alpaca_format
    data_info_path = save2json(data_info_json, data_info_path)
    logger.info('data_info_path is {}', data_info_path)


def create_data_import_box(dataset_dir: "gr.Textbox", dataset: "gr.Dropdown") -> Dict[str, "Component"]:
    """
    data import box
    """
    data_import_btn = gr.Button('导入数据', interactive=True, scale=1, )
    with gr.Column(visible=False, elem_classes="modal-box") as import_data_box:
        inputs = gr.components.File(label="上传excel文件")
        submit_btn = gr.Button('确定')
        submit_btn.click(import_data, [dataset_dir, dataset, inputs])
        data_import_close_btn = gr.Button('关闭')

    data_import_btn.click(lambda: gr.Column(visible=True), outputs=[import_data_box], queue=False)
    data_import_close_btn.click(lambda: gr.Column(visible=False), outputs=[import_data_box], queue=False)
    return dict(
        data_import_btn=data_import_btn,
        data_import_close_btn=data_import_close_btn
    )
