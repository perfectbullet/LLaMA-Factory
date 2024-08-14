import os
import matplotlib

from llamafactory.webui.interface import create_ui

matplotlib.use('TkAgg')


def main():
    gradio_share = os.environ.get("GRADIO_SHARE", "0").lower() in ["true", "1"]
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = os.environ.get("GRADIO_SERVER_PORT", 7871)
    create_ui().queue().launch(
        share=gradio_share,
        server_name=server_name,
        server_port=server_port,
        inbrowser=True,
        show_error=True
    )


if __name__ == "__main__":
    main()
