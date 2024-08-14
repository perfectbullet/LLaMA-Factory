from typing import Optional, List

from bson import ObjectId
from pydantic import ConfigDict, BaseModel, Field, EmailStr
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated

'''
为了避免 SQLAlchemy*模型*和 Pydantic*模型*之间的混淆，
我们将有models.py（SQLAlchemy 模型的文件）和schemas.py（ Pydantic 模型的文件）。
这些 Pydantic 模型或多或少地定义了一个“schema”（一个有效的数据形状）。
'''

# Represents an ObjectId field in the database.
# It will be represented as a `str` on the model so that it can be serialized to JSON.
PyObjectId = Annotated[str, BeforeValidator(str)]


class LLMBaseModel(BaseModel):
    """
    Container for a single LLMBase record.
    """
    # The primary key for the LLMBaseModel, stored as a `str` on the instance.
    # This will be aliased to `_id` when sent to MongoDB,
    # but provided as `id` in the API requests and responses.
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    modelname: str = Field(...)
    template: str = Field(...)
    modelpath: str = Field(...)
    description: str = Field(...)

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                'modelname': 'LLaMA3-8B-Chinese-Chat',
                 "template": "llama3",
                 "modelpath": "./models/LLaMA3-8B-Chinese-Chat",
                 'description': '首个中文微调LLaMa 3模型，2222基于 Meta-Llama-3-8B-Instruct 模型',
            }
        },
    )


class UpdateLLMBaseModel(BaseModel):
    """
    A set of optional updates to be made to a document in the database.
    """
    modelname: str = Field(...)
    template: str = Field(...)
    modelpath: str = Field(...)
    description: str = Field(...)
    model_config = ConfigDict(
        # arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
        json_schema_extra={
            "example": {
                'modelname': 'LLaMA3-8B-Chinese-Chat',
                 "template": "llama3",
                 "modelpath": "./models/LLaMA3-8B-Chinese-Chat",
                 'description': '首个中文微调LLaMa 3模型，基于 Meta-Llama-3-8B-Instruct 模型',
            }
        },
    )


class LLMBaseCollection(BaseModel):
    """
    A container holding a list of `LLMBaseModel` instances.

    This exists because providing a top-level array in a JSON response can be a [vulnerability](https://haacked.com/archive/2009/06/25/json-hijacking.aspx/)
    """

    LLMBases: List[LLMBaseModel]
