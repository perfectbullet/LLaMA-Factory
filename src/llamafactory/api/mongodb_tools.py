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
    listed_indexes = dataset_info_collection.list_indexes()
    async for index in listed_indexes:
        logger.info('index is {}'.format(index))
    try:
        re1 = await dataset_info_collection.create_index('dataset_name', unique=True)
        logger.info('init mongodb re1 IS {}'.format(re1))
    except Exception as e:
        logger.error('dataset_info_collection.create_index error is {}'.format(e))
    try:
        re2 = await dataset_info_collection.create_index('file_name', unique=True)
        logger.info('init mongodb re2 IS {}'.format(re2))
    except Exception as e:
        logger.error('dataset_info_collection.create_index error is {}'.format(e))
    logger.info("init mongodb finished")


# # 必须要在其他的线程中执行，不然回影响到fastapi 线程执行
# asyncio.run(init_mongodb())


# loop = asyncio.new_event_loop()
# # asyncio.run_coroutine_threadsafe(init_mongodb(), loop)
# task = loop.create_task(init_mongodb())
# # print(task)
# loop.run_until_complete(task)

logger.info('MONGODB LLMBase_collection IS {}'.format(llmbase_collection))
