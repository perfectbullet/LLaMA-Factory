import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

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
from ..extras.constants import API_SUPPORTED_MODELS
from ..extras.logging import get_logger
from ..extras.misc import torch_gc
from ..extras.packages import is_fastapi_available, is_starlette_available, is_uvicorn_available

logger = get_logger(__name__)

if is_fastapi_available():
    from fastapi import Depends, FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
    from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
    from fastapi import FastAPI, applications
    from fastapi.staticfiles import StaticFiles

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
        os.environ['LANG']="zh_CN.UTF-8"
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
        os.environ['LANG']="zh_CN.UTF-8"
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
        os.environ['LANG']="zh_CN.UTF-8"
        # 这里固定一下三个参数 export LANG="zh_CN";export LANGUAGE="zh_CN";export LC_ALL="zh_CN";
        if request.model in ['GX-7B-Chat-5000B', 'Qwen-7B']:
            request.temperature = 0.3
            request.top_p = 0.9
            request.max_tokens = 512
        elif request.model in ['GX-8B-Chinese-Chat-zhaobiao',]:
            request.temperature = 0.6
            request.top_p = 0.7
            request.max_tokens = 1024
        elif request.model in ['GX-8B-Chinese-Chat-gjb5000b',]:
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

    return app


def run_api() -> None:
    chat_model = ApiChatModel()
    app = create_app(chat_model)
    api_host = os.environ.get("API_HOST", "0.0.0.0")
    api_port = int(os.environ.get("API_PORT", "8010"))
    print("Visit  http://localhost:{}/docs for API document."
          "\nVisit  http://localhost:{}/redoc for API document.".format(api_port, api_port))
    uvicorn.run(app, host=api_host, port=api_port)
