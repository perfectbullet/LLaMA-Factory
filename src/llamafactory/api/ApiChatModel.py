from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Generator, List, Optional, Sequence
import time

from ..chat import ChatModel
from ..extras.misc import torch_gc

class ApiChatModel(ChatModel):
    def __init__(self, args: Optional[Dict[str, Any]] = None, lazy_init: bool = True) -> None:
        self.engine: Optional["BaseEngine"] = None
        if not lazy_init:
            # read arguments from command line
            super().__init__(args)

    def load_from_args(self, args: Optional[Dict[str, Any]] = None) -> None:
        self.unload_from_api()
        time.sleep(5)
        super().__init__(args)

    def unload_from_api(self):
        self.engine = None
        torch_gc()
        time.sleep(5)
        
    @property
    def loaded(self) -> bool:
        return self.engine is not None
