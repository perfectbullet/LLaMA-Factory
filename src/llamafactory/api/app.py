import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from bson import ObjectId
from bson.errors import InvalidId
from loguru import logger
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError
from typing_extensions import Annotated

from .ApiChatModel import ApiChatModel
from .chat import (
    create_chat_completion_response,
    create_score_evaluation_response,
    create_stream_chat_completion_response,
)
from .protocol import (
    Role,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelCardDes,
    ModelCard,
    ModelDesList,
    ModelList,
    ScoreEvaluationRequest,
    ScoreEvaluationResponse,
    ChatCompletionResponseChoice,
    ChatCompletionMessage,
    Finish,
    ChatCompletionResponseUsage,
    MODEL_STOPPED,
    MODEL_RUNNING,
)
from .mongodb_tools import llmbase_collection, dataset_info_collection, db, init_mongodb

from .schemas import LLMBaseModel, LLMBaseCollection, UpdateLLMBaseModel, DataSetInfo, DataSetInfoList, \
    DataSetFormatList, DataSetFormat, FinetuningArgs, FinetuningArgsList
from ..extras.constants import API_SUPPORTED_MODELS
from ..extras.misc import torch_gc
from ..extras.packages import is_fastapi_available, is_starlette_available, is_uvicorn_available

if is_fastapi_available():
    from fastapi import Depends, FastAPI, HTTPException, status, Body
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
    from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
    from fastapi import FastAPI, applications
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import Response

if is_starlette_available():
    from sse_starlette import EventSourceResponse

if is_uvicorn_available():
    import uvicorn


@asynccontextmanager
async def lifespan(app: "FastAPI"):  # collects GPU memory
    yield
    torch_gc()


def create_app(chat_model: ApiChatModel) -> FastAPI:
    def swagger_ui_patch(*args, **kwargs):
        return get_swagger_ui_html(
            *args, **kwargs,
            swagger_js_url='/statics/swagger-ui/swagger-ui-bundle.js',
            swagger_css_url='/statics/swagger-ui/swagger-ui.css',
            swagger_favicon_url='/statics/swagger-ui/favicon.png',
        )

    def redoc_ui_path(*args, **kwargs):
        return get_redoc_html(
            *args,
            **kwargs,
            redoc_js_url='/statics/swagger-ui/redoc.standalone.js',
            redoc_favicon_url='/statics/swagger-ui/favicon.png',
        )

    applications.get_swagger_ui_html = swagger_ui_patch
    applications.get_redoc_html = redoc_ui_path

    app = FastAPI(
        title="观想科技AI小组的接口",
        lifespan=lifespan,
        description="观想科技AI小组的fastapi接口"
    )

    # 挂载静态文件
    app.mount('/statics', StaticFiles(directory='./swagger_statics'), name='statics')

    # 跨域设置
    origins = [
        "*"
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    api_key = os.environ.get("API_KEY")
    # logger.info("==== api_key ====\n{}\n\n".format(api_key))
    security = HTTPBearer(auto_error=False)
    logger.info("============ API_SUPPORTED_MODELS ============ {} \n\n".format(API_SUPPORTED_MODELS))
    # logger.info("============ API_SUPPORTED_MODELS ============ {} \n\n".format(API_SUPPORTED_MODELS))
    # API_SUPPORTED_MODELS['LLaMA3-8B-Chinese-Chat']['status'] = "running"
    logger.info("============ API_SUPPORTED_MODELS after change ============ {} \n\n".format(API_SUPPORTED_MODELS))

    async def verify_api_key(auth: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)]):
        if api_key and (auth is None or auth.credentials != api_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")

    @app.post(
        '/v1/load_model',
        response_model=ModelCard,
        dependencies=[Depends(verify_api_key)],
    )
    async def load_model(request: ModelCard):
        """
        a load model api
        """
        # 设置环境变量
        os.environ['LANG'] = "zh_CN.UTF-8"
        logger.info('load model, model card is {}'.format(request))
        model_args = API_SUPPORTED_MODELS[request.id]['model_args']
        logger.info('model_args is {}'.format(model_args))
        for k, value in API_SUPPORTED_MODELS.items():
            # 直接把所有状态改掉
            value['status'] = MODEL_STOPPED
        API_SUPPORTED_MODELS[request.id]['status'] = MODEL_RUNNING
        chat_model.load_from_args(args=model_args)
        res_model = ModelCard(id=request.id)
        return res_model

    @app.post(
        '/v1/unload_model',
        response_model=ModelCardDes,
        dependencies=[Depends(verify_api_key)],
    )
    async def unload_model(request: ModelCard):
        """
        unload model api
        直接卸载所有模型
        """
        model_id = request.id
        # 设置环境变量
        os.environ['LANG'] = "zh_CN.UTF-8"
        for k, value in API_SUPPORTED_MODELS.items():
            value['status'] = MODEL_STOPPED
        logger.info('unload_model ModelCard is {}'.format(request))
        if chat_model.engine is None:
            return ModelCardDes(
                id=request.id,
                status=MODEL_STOPPED,
                description=API_SUPPORTED_MODELS[request.id]['description']
            )
        logger.info('unload_from_api start')
        chat_model.unload_from_api()
        logger.info('unload_from_api finished')
        res_model = ModelCardDes(
            id=request.id,
            status=MODEL_STOPPED,
            description=API_SUPPORTED_MODELS[request.id]['description']
        )
        return res_model

    @app.get(
        "/v1/models",
        response_model=ModelDesList,
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(verify_api_key)],
    )
    async def list_models():
        # a new models api
        available_models = []
        for model_name, model_info in API_SUPPORTED_MODELS.items():
            model_card = ModelCardDes(id=model_name, description=model_info['description'], status=model_info['status'])
            print('model_name: {}, model_info: {}'.format(model_name, model_info))
            available_models.append(model_card)
        return ModelList(data=available_models)

    @app.post(
        "/v1/chat/completions",
        response_model=ChatCompletionResponse,
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(verify_api_key)],
    )
    async def create_chat_completion(request: ChatCompletionRequest):
        logger.info('request is %s', request)
        logger.info('LANG is %s', os.environ['LANG'])
        # 设置环境变量
        os.environ['LANG'] = "zh_CN.UTF-8"
        # 这里固定一下三个参数 export LANG="zh_CN";export LANGUAGE="zh_CN";export LC_ALL="zh_CN";
        if request.model in ['GX-7B-Chat-5000B', 'Qwen-7B']:
            request.temperature = 0.3
            request.top_p = 0.9
            request.max_tokens = 512
        elif request.model in ['GX-8B-Chinese-Chat-zhaobiao', ]:
            request.temperature = 0.6
            request.top_p = 0.7
            request.max_tokens = 1024
        elif request.model in ['GX-8B-Chinese-Chat-gjb5000b', ]:
            request.temperature = 0.5
            request.top_p = 0.8
            request.max_tokens = 512
        else:
            request.temperature = 0.5
            request.top_p = 0.9
            request.max_tokens = 512
        logger.info('new request is %s', request)
        if chat_model.engine is None:
            # 模型未加载
            completion_id = "chatcmpl-{}".format(uuid.uuid4().hex)
            response_message = ChatCompletionMessage(role=Role.ASSISTANT, content='model not load', tool_calls=None)
            finish_reason = Finish.STOP
            choice = ChatCompletionResponseChoice(
                index=0,
                message=response_message,
                finish_reason=finish_reason
            )
            usage = ChatCompletionResponseUsage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30
            )

            res = ChatCompletionResponse(
                id=completion_id,
                model=request.model,
                choices=[choice, ],
                usage=usage
            )
            return res

        if not chat_model.engine.can_generate:
            raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="Not allowed")

        if request.stream:
            generate = create_stream_chat_completion_response(request, chat_model)
            # return EventSourceResponse(['ok{}'.format(i) for i in range(20)], media_type="text/event-stream")
            return EventSourceResponse(generate, media_type="text/event-stream")
        else:
            return await create_chat_completion_response(request, chat_model)

    @app.post(
        "/v1/score/evaluation",
        response_model=ScoreEvaluationResponse,
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(verify_api_key)],
    )
    async def create_score_evaluation(request: ScoreEvaluationRequest):
        if chat_model.engine.can_generate:
            raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="Not allowed")

        return await create_score_evaluation_response(request, chat_model)

    # ****************************** mongodb初始化 ******************************
    @app.put(
        "/v1/super/init_mongodb/{token}",
        response_model=str,
        response_description="mongodb初始化",
        summary='mongodb初始化'
    )
    async def init_mongodb_super(token: str):
        """
        mongodb初始化
        """
        logger.info('mongodb初始化 %s', token)
        if token == '121531845':
            await init_mongodb()
            return 'ok'
        else:
            return 'token is not ok'

    # ****************************** 基础语言模型相关接口 ******************************
    @app.post(
        "/v1/finetuning/base_models/",
        response_description="基础语言模型",
        response_model=LLMBaseModel,
        status_code=status.HTTP_201_CREATED,
        response_model_by_alias=True,
        summary='新增基础模型配置'
    )
    async def create_llmbase(llmbase: LLMBaseModel = Body(...)):
        """
        插入新的 LLMBase 记录。
        将创建一个唯一的“id”并在响应中提供。
        """
        logger.info('create_llmbase %s', create_llmbase)
        new_llmbase = await llmbase_collection.insert_one(
            llmbase.model_dump(by_alias=True, exclude=["id"])
        )
        created_llmbase = await llmbase_collection.find_one(
            {"_id": new_llmbase.inserted_id}
        )
        return created_llmbase

    @app.get(
        "/v1/finetuning/base_models/",
        response_description="基础模型信息列表",
        response_model=LLMBaseCollection,
        response_model_by_alias=False,
        summary='获取所有基础模型信息的列表'
    )
    async def list_llmbase():
        # 这个注释会被写道api文档中
        # 函数名 list_LLMBases 也会以 list LLMBases 写到接口文档中
        """
        列出数据库中的所有 LLMBase 数据
        响应未分页且仅限于 1000 个结果。
        """
        return LLMBaseCollection(llm_bases_list=await llmbase_collection.find().to_list(1000))

    @app.get(
        "/v1/finetuning/base_models/{model_id}",
        response_description="单个 LLMBase",
        response_model=LLMBaseModel,
        response_model_by_alias=False,
        summary='获取单个 LLMBase'
    )
    async def show_llmbase(model_id: str):
        """
        获取特定 LLMBase 的记录，按 id 查找。
        """
        if (
                llmbase := await llmbase_collection.find_one({"_id": ObjectId(model_id)})
        ) is not None:
            return llmbase
        raise HTTPException(status_code=404, detail=f"LLMBase {model_id} not found")

    @app.put(
        "/v1/finetuning/base_models/{model_id}",
        response_description="修改一个 LLMBase",
        response_model=LLMBaseModel,
        response_model_by_alias=False,
        summary='修改一个 LLMBase'
    )
    async def update_llmbase(model_id: str, llmbase: UpdateLLMBaseModel = Body(...)):
        """
        更新现有 LLMBase 记录的各个字段。 仅更新提供的字段。任何缺失或空字段都将被忽略。
        """
        llmbase = {
            k: v for k, v in llmbase.model_dump(by_alias=True).items() if v is not None
        }
        if len(llmbase) >= 1:
            update_result = await llmbase_collection.find_one_and_update(
                {"_id": ObjectId(model_id)},
                {"$set": llmbase},
                return_document=ReturnDocument.AFTER,
            )
            if update_result is not None:
                return update_result
            else:
                raise HTTPException(status_code=404, detail=f"LLMBase {model_id} not found")
        # The update is empty, but we should still return the matching document:
        if (existing_LLMBase := await llmbase_collection.find_one({"_id": model_id})) is not None:
            return existing_LLMBase
        raise HTTPException(status_code=404, detail=f"LLMBase {model_id} not found")

    @app.delete(
        "/v1/finetuning/base_models/{model_id}",
        response_description="删除一条",
        summary='删除一条 LLMBase'
    )
    async def delete_llmbase(model_id: str):
        """
        从数据库中删除一条 LLMBase 记录
        """
        delete_result = await llmbase_collection.delete_one({"_id": ObjectId(model_id)})

        if delete_result.deleted_count == 1:
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        raise HTTPException(status_code=404, detail=f"LLMBase {model_id} not found")

    # ****************************** 数据集相关接口 ******************************
    @app.post(
        '/v1/finetuning/create_dataset',
        # response_description 响应信息描述
        response_description='当前创建的一条数据集信息',
        # 接口的概括
        summary='增加一个数据集信息',
        status_code=status.HTTP_201_CREATED,
        # 接口返回的数据模式
        response_model=DataSetInfo,
    )
    async def create_dataset(dataset_info: DataSetInfo):
        """
        模型微调数据集数据信息
        """
        logger.info('create dataset %s', dataset_info)
        dinfo = dataset_info.model_dump(exclude={'id', })
        logger.info('dinfo %s', dinfo)

        try:
            new_dataset_info = await dataset_info_collection.insert_one(dinfo)
        except DuplicateKeyError as e:
            detail = "DuplicateKeyError, {}".format(e)
            logger.error(detail)
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)

        created_dinfo = await dataset_info_collection.find_one(
            {"_id": new_dataset_info.inserted_id}
        )
        return created_dinfo

    @app.get(
        '/v1/finetuning/dataset_infos',
        response_description='数据集信息列表',
        response_model=DataSetInfoList,
        summary='获取所有数据集信息列表'
     )
    async def get_datasets():
        """
        获取所有数据集信息列表
        响应未分页且仅限于 1000 个结果
        """
        return DataSetInfoList(dataset_info_list=await dataset_info_collection.find().to_list(1000))

    @app.put(
        "/v1/finetuning/dataset_infos/{info_id}",
        response_description="修改一个数据集信息",
        response_model=DataSetInfo,
        response_model_by_alias=False,
        summary='修改一个数据集信息'
    )
    async def update_llmbase(info_id: str, dataset_info: DataSetInfo = Body(...)):
        """
        更新现有 LLMBase 记录的各个字段。 仅更新提供的字段。任何缺失或空字段都将被忽略。
        """
        dataset_info = {
            k: v for k, v in dataset_info.model_dump(by_alias=True).items() if v is not None
        }
        if len(dataset_info) >= 1:
            update_result = await dataset_info_collection.find_one_and_update(
                {"_id": ObjectId(info_id)},
                {"$set": dataset_info},
                return_document=ReturnDocument.AFTER,
            )
            if update_result is not None:
                return update_result
            else:
                raise HTTPException(status_code=404, detail=f"dataset_info {info_id} not found")
        # The update is empty, but we should still return the matching document:
        if (existing_student := await dataset_info_collection.find_one({"_id": info_id})) is not None:
            return existing_student
        raise HTTPException(status_code=404, detail=f"dataset_info {info_id} not found")

    # ****************************** 数据集格式相关接口 ******************************
    @app.post(
        '/v1/finetuning/create_dataset_format',
        response_description='当前创建的一条数据集格式信息',
        summary='增加一个数据集格式信息',
        status_code=status.HTTP_201_CREATED,
        response_model=DataSetFormat,
    )
    async def create_dataset_format(dataset_format: DataSetFormat):
        """
        模型微调数据集数据信息
        """
        logger.info('create dataset_format %s', dataset_format)
        finfo = dataset_format.model_dump(exclude={'id', })
        logger.info('finfo %s', finfo)
        collection = db.get_collection("dataset_format")
        try:
            new_format_info = await collection.insert_one(finfo)
        except DuplicateKeyError as e:
            detail = "DuplicateKeyError, {}".format(e)
            logger.error(detail)
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)

        created_dinfo = await collection.find_one(
            {"_id": new_format_info.inserted_id}
        )
        return created_dinfo

    @app.get(
        '/v1/finetuning/dataset_format',
        response_description='数据集格式',
        response_model=DataSetFormatList,
        summary='获取数据集格式'
     )
    async def get_dataset_format():
        """
        获取数据集格式
        """
        collection = db.get_collection("dataset_format")
        res = await collection.find().to_list(100)
        dataset_format_list = DataSetFormatList(dataset_format_list=res)
        return dataset_format_list

    # ****************************** 微调相关接口 ******************************
    @app.post(
        '/v1/finetuning/run_finetuning',
        response_description='微调状态返回',
        response_model=FinetuningArgs,
        status_code=status.HTTP_201_CREATED,
        summary='模型微调接口',
    )
    async def run_finetuning(fine_tuning_args: FinetuningArgs):
        # logger.info('fine_tuning_args %s', fine_tuning_args)
        fargs = fine_tuning_args.model_dump(exclude={'id', })
        logger.info('fargs %s', fargs)
        collection = db.get_collection("fine_tuning_args")
        try:
            new_fargs = await collection.insert_one(fargs)
        except DuplicateKeyError as e:
            detail = "DuplicateKeyError, {}".format(e)
            logger.error(detail)
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)
        created_fargs = await collection.find_one(
            {"_id": new_fargs.inserted_id}
        )
        return created_fargs

    @app.get(
        '/v1/finetuning/get_finetuning_args',
        response_description='微调参数列表',
        response_model=FinetuningArgsList,
        summary='微调参数列表'
    )
    async def list_finetuning_args():
        """
        获取数据集格式
        """
        collection = db.get_collection("fine_tuning_args")
        res = await collection.find().to_list(100)
        fine_tuning_args_list = FinetuningArgsList(fine_tuning_args_list=res)
        return fine_tuning_args_list

    @app.get(
        "/v1/finetuning/get_finetuning_args/{args_id}",
        response_description="单个 微调参数",
        response_model=FinetuningArgs,
        summary='获取单个 微调参数'
    )
    async def show_finetuning_args(args_id: str):
        """
        获取特定 微调参数 的记录，按 id 查找。
        """
        collection = db.get_collection("fine_tuning_args")
        if (
            finetuning_args := await collection.find_one({"_id": ObjectId(args_id)})
        ) is not None:
            return finetuning_args
        raise HTTPException(status_code=404, detail=f"finetuning_args {args_id} not found")

    @app.delete("/v1/finetuning/finetuning_args/{args_id}", response_description="清理一个 微调参数")
    async def delete_finetuning_args(args_id: str):
        """
        清理一个 微调参数
        """
        collection = db.get_collection("fine_tuning_args")
        try:
            delete_result = await collection.delete_one({"_id": ObjectId(args_id)})
        except Exception as e:
            detail = "InvalidId, {}".format(e)
            logger.error(detail)
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)
        if delete_result.deleted_count == 1:
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        raise HTTPException(status_code=404, detail=f"finetuning_args {args_id} 没有找到这个")

    return app


def run_api() -> None:
    chat_model = ApiChatModel()
    app = create_app(chat_model)
    api_host = os.environ.get("API_HOST", "0.0.0.0")
    api_port = int(os.environ.get("API_PORT", "8010"))
    print("Visit  http://localhost:{}/docs for API document."
          "\nVisit  http://localhost:{}/redoc for API document.".format(api_port, api_port))
    uvicorn.run(app, host=api_host, port=api_port)
