import os

from .common import save_config
from .components import (
    create_chat_box,
    create_eval_tab,
    create_export_tab,
    create_infer_tab,
    create_top,
    create_train_tab,
)
from .css import CSS
from .engine import Engine
from ..extras.packages import is_gradio_available

if is_gradio_available():
    import gradio as gr


def create_ui(demo_mode: bool = False) -> gr.Blocks:
    engine = Engine(demo_mode=demo_mode, pure_chat=False)
    
    
    with gr.Blocks(title="观想科技 LLaMA Board", css=CSS) as demo:
        #gr.HTML("<h1><center>观想科技 LLaMA Board</center></h1>")
        # gr.Image('./assets/head_01.png')
        # gr.HTML("<img src='/file=./assets/head_01.png'>")
        # image_path = "static/images/head_01.png"
        gr.HTML(f"""<img src="/file=static/images/head_01.png">""")
    
        with gr.Row():
            with gr.Column(scale=9):
                engine.manager.add_elems("top", create_top())
                lang: "gr.Dropdown" = engine.manager.get_elem_by_id("top.lang")
        
                with gr.Tab("训练"):
                    engine.manager.add_elems("train", create_train_tab(engine))
        
                with gr.Tab("评估"):
                    engine.manager.add_elems("eval", create_eval_tab(engine))
        
                with gr.Tab("对话"):
                    engine.manager.add_elems("infer", create_infer_tab(engine))
        
                if not demo_mode:
                    with gr.Tab("导出发布"):
                        engine.manager.add_elems("export", create_export_tab(engine))
        
                demo.load(engine.resume, outputs=engine.manager.get_elem_list(), concurrency_limit=None)
                lang.change(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)
                lang.input(save_config, inputs=[lang], queue=False)
            with gr.Column(scale=1):
                # gr.Image('./assets/help_04.png')
                # gr.Image('./assets/help_05.png')
                gr.HTML(f"""<img src="/file=static/images/help_05.png">""")
                # gr.HTML("<h3><center>帮助中心</center></h3>")
                # gr.HTML("<h4>教程</h4>")
                # gr.HTML("<p>这里写教程内容1.模型选择2.数据集选择3。模型训练。。。。</p>")
                btn = gr.Button("训练简介", link="/file=static/images/2024模型训练综合信息处理平台.pdf")
                gr.Video("./assets/model_train.mp4")
    return demo


def create_web_demo() -> gr.Blocks:
    engine = Engine(pure_chat=True)

    with gr.Blocks(title="Web Demo", css=CSS) as demo:
        lang = gr.Dropdown(choices=["en", "zh"])
        engine.manager.add_elems("top", dict(lang=lang))

        _, _, chat_elems = create_chat_box(engine, visible=True)
        engine.manager.add_elems("infer", chat_elems)

        demo.load(engine.resume, outputs=engine.manager.get_elem_list(), concurrency_limit=None)
        lang.change(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)
        lang.input(save_config, inputs=[lang], queue=False)

    return demo


def run_web_ui() -> None:
    gradio_share = os.environ.get("GRADIO_SHARE", "0").lower() in ["true", "1"]
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    create_ui().queue().launch(share=gradio_share, server_name=server_name, inbrowser=True)


def run_web_demo() -> None:
    gradio_share = os.environ.get("GRADIO_SHARE", "0").lower() in ["true", "1"]
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    create_web_demo().queue().launch(share=gradio_share, server_name=server_name, inbrowser=True)
