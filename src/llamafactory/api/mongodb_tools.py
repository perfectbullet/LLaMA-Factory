import os

import motor.motor_asyncio
from loguru import logger


MONGODB_URL = os.environ["MONGODB_URL"]
logger.info('MONGODB_URL IS {}'.format(MONGODB_URL))

client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
# 指定数据库
db = client.llamafactory_data
logger.info('MONGODB db IS {}'.format(db))
# 指定数据库指定集合
llmbase_collection = db.get_collection("LLMBase")
logger.info('MONGODB LLMBase_collection IS {}'.format(llmbase_collection))
