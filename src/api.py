import os

import uvicorn
from loguru import logger

from llamafactory.api.app import create_app
from llamafactory.api.engine import ApiEngine

from llamafactory.api.app import run_api

#
# def main():
#     logger.info("********\n os.getcwd {}\n\n********".format(os.getcwd()))
#     engine = ApiEngine()
#     app = create_app(engine)
#     api_host = os.environ.get("API_HOST", "0.0.0.0")
#     api_port = os.environ.get("API_PORT", 8010)
#     logger.info("Visit  http://localhost:{}/docs for API document."
#                 "\nVisit  http://localhost:{}/redoc for API document.".format(api_port, api_port))
#     uvicorn.run(app, host=api_host, port=api_port)


if __name__ == "__main__":
    run_api()
