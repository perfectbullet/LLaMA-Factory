from .apirunner import ApiRunner
from ..webui.utils import create_ds_config


class ApiEngine:
    def __init__(self, demo_mode: bool = False) -> None:
        self.demo_mode = demo_mode
        self.runner = ApiRunner(demo_mode)
        if not demo_mode:
            create_ds_config()
