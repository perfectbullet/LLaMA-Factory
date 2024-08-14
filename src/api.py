import os

import uvicorn

from llamafactory.api.app import create_app
# from llamafactory.chat import ChatModel
from llamafactory.api.ApiChatModel import ApiChatModel
# from llamafactory.extras.logging import get_logger
from loguru import logger


def main():
    logger.info("********\n os.getcwd {}\n\n********".format(os.getcwd()))

    chat_model = ApiChatModel()
    app = create_app(chat_model)
    api_host = os.environ.get("API_HOST", "0.0.0.0")
    api_port = os.environ.get("API_PORT", 8010)
    logger.info("Visit  http://localhost:{}/docs for API document."
                "\nVisit  http://localhost:{}/redoc for API document.".format(api_port, api_port))
    uvicorn.run(app, host=api_host, port=api_port)


if __name__ == "__main__":
    main()
