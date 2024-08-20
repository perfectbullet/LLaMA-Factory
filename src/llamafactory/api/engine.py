from .apirunner import ApiRunner
from .ApiChatModel import ApiChatModel

from ..webui.utils import create_ds_config


class ApiEngine:
    def __init__(self, demo_mode: bool = False) -> None:
        self.demo_mode = demo_mode
        # 训练器
        self.runner = ApiRunner(demo_mode)
        # 模型
        self.chat_model = ApiChatModel()
        if not demo_mode:
            create_ds_config()
