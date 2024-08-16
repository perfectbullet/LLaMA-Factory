from typing import Optional, List, Literal
from enum import Enum, unique

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

@unique
class AllowedFormat(str, Enum):
    ALPACA = 'alpaca'
    C4 = 'c4'

class DataSetInfo(BaseModel):
    """
    DataSetInfo LLM模型微调数据集数据信息
    """
    # Field(...)   参考文档  https://fastapi.tiangolo.com/zh/tutorial/query-params-str-validations/#_6
    # 使用省略号(...)声明必需参数
    # 你可以声明一个参数可以接收None值，但它仍然是必需的。这将强制客户端发送一个值，即使该值是None。
    # 如果你觉得使用 ... 不舒服，你也可以从 Pydantic 导入并使用 Required： 2.8版本废除了
    # 请记住，在大多数情况下，当你需要某些东西时，可以简单地省略 default 参数，因此你通常不必使用 ... 或 Required 如： Field()
    id: Optional[PyObjectId] = Field(alias='_id', default=None)
    # title, description, 限定dataset_format取值 会在 http://localhost:8010/redoc 中展示出来
    dataset_name: str = Field(title="数据集名称", description="数据集名称", max_length=300,)
    # AllowedFormat 限定dataset_format取值， 还有一种写法是 dataset_format: Literal["alpaca", "c4"]
    # dataset_format2: Literal["alpaca", "c4"] = Field(default=..., title="title数据集格式", description="description数据集格式")
    dataset_format: AllowedFormat = Field(title="数据集格式", description="数据集格式,表示数据集如何构成,有那些字段")
    dataset_description: str = Field(title="数据集描述", description="描述数据集来源，特点，适合什么样的微调方式")
    file_name: str = Field(title="数据集文件名", description="数据集文件名")
    model_config = ConfigDict(
        json_schema_extra={
            'example': {
                'dataset_name': 'alpaca_zh_demo',
                'dataset_format': 'alpaca',
                'dataset_description': '一个中文的alpaca数据集示例',
                'file_name': 'alpaca_zh_demo.json'
            }
        }
    )


class DataSetInfoList(BaseModel):
    """
    DataSetInfoList 是 DataSetInfo 的列表
    """
    dataset_info_list: List[DataSetInfo]


class DataSetFormat(BaseModel):
    """
    DataSetFormat 数据集格式
    """
    id: Optional[PyObjectId] = Field(alias='_id', default=None)
    dataset_format: str = Field(title="数据集格式名称", description="数据集格式名称", max_length=30,)
    dataset_description: str = Field(title="数据集描述", description="数据集格式描述")
    model_config = ConfigDict(
        json_schema_extra={
            'example': {
                'dataset_format': 'alpaca',
                'dataset_description': 'alpaca格式数据集是一种指令微调数据集',
            }
        }
    )


class DataSetFormatList(BaseModel):
    """
    DataSetFormatList 是 格式 的列表
    """
    dataset_format_list: List[DataSetFormat]


class UpdateDataSetInfo(BaseModel):
    """
    UpdateDataSetInfo 更新LLM模型微调数据集数据信息
    id 在url中显示
    """
    dataset_name: str = Field(title="数据集名称", description="数据集名称", max_length=300,)
    dataset_format: AllowedFormat = Field(title="数据集格式", description="数据集格式,表示数据集如何构成,有那些字段")
    dataset_description: str = Field(title="数据集描述", description="描述数据集来源，特点，适合什么样的微调方式")
    file_name: str = Field(title="数据集文件名", description="数据集文件名")
    model_config = ConfigDict(
        json_schema_extra={
            'example': {
                'dataset_name': 'alpaca_zh_demo',
                'dataset_format': 'alpaca',
                'dataset_description': '一个中文的alpaca数据集示例',
                'file_name': 'alpaca_zh_demo.json'
            }
        }
    )


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

    llm_bases_list: List[LLMBaseModel]
