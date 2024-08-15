import os
import asyncio

import motor.motor_asyncio

from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from loguru import logger


MONGODB_URL = os.environ["MONGODB_URL"]
logger.info('MONGODB_URL IS {}'.format(MONGODB_URL))

client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
# 指定数据库
db: AsyncIOMotorDatabase = client.llamafactory_data
logger.info('MONGODB db IS {}'.format(db))
# 指定数据库指定集合
llmbase_collection: AsyncIOMotorCollection = db.get_collection("LLMBase")
dataset_info_collection: AsyncIOMotorCollection = db.get_collection("datasets_info")


async def init_mongodb():
    logger.info("init mongodb started")
    re1 = await asyncio.run(await dataset_info_collection.create_index('dataset_name', unique=True))
    re2 = await asyncio.run(await dataset_info_collection.create_index('file_name', unique=True))
    logger.info('init mongodb re1 IS {}'.format(re1))
    logger.info('init mongodb re2 IS {}'.format(re2))

    logger.info("init mongodb finished")


# 必须要在其他的线程中执行，不然回影响到fastapi 线程执行
loop = asyncio.get_event_loop()
asyncio.run_coroutine_threadsafe(init_mongodb(), loop)

logger.info('MONGODB LLMBase_collection IS {}'.format(llmbase_collection))
