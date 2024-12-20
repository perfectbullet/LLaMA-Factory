import os
# import matplotlib

os.environ["HTTP_PROXY"] = ''
os.environ["HTTPS_PROXY"] = ''
os.environ["all_proxy"] = ''
os.environ["ALL_PROXY"] = ''

import gradio as gr
from llamafactory.webui.interface import create_ui

# error in docker container
# matplotlib.use('TkAgg')


def main():
    gradio_share = os.environ.get("GRADIO_SHARE", "0").lower() in ["true", "1"]
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", 7871))
    gr.set_static_paths(paths=["static/images/"])
    create_ui().queue().launch(
        share=gradio_share,
        server_name=server_name,
        server_port=server_port,
        inbrowser=True,
        show_error=True
    )


if __name__ == "__main__":
    main()
