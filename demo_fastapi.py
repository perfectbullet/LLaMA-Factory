import os

import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel

from loguru import logger


# 设置不同级别的日志输出文件
logger.add("debug.log", level="DEBUG", rotation="10 MB", filter=lambda record: record["level"].name == "DEBUG")
logger.add("info.log", level="INFO", rotation="10 MB", filter=lambda record: record["level"].name == "INFO")
logger.add("warning.log", level="WARNING", rotation="10 MB", filter=lambda record: record["level"].name == "WARNING")
logger.add("error.log", level="ERROR", rotation="10 MB", filter=lambda record: record["level"].name == "ERROR")
logger.add("critical.log", level="CRITICAL", rotation="10 MB", filter=lambda record: record["level"].name == "CRITICAL")



class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Foo",
                    "description": "A aaaaaaaaaaaaaaa very nice Item",
                    "price": 35.4,
                    "tax": 3.2,
                }
            ]
        }
    }


def create_app():
    app = FastAPI()
    @app.put("/items/{item_id}")
    async def update_item(item_id: int, item: Item):
        results = {"item_id": item_id, "item": item}
        return results
    return app


def main():
    logger.info("********\n os.getcwd {}\n\n********".format(os.getcwd()))
    app = create_app()
    api_host = os.environ.get("API_HOST", "0.0.0.0")
    api_port = int(os.environ.get("API_PORT", "8011"))
    logger.info("Visit http://localhost:{}/docs for API document.".format(api_port))
    uvicorn.run(app, host=api_host, port=api_port)


if __name__ == "__main__":
    main()

