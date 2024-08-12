from fastapi.middleware.cors import CORSMiddleware

from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi import FastAPI, applications

from fastapi.staticfiles import StaticFiles

from PaddleOCRFastAPI.routers import ocr


def swagger_ui_patch(*args, **kwargs):
    return get_swagger_ui_html(
        *args, **kwargs,
        swagger_js_url='/statics/swagger-ui/swagger-ui-bundle.js',
        swagger_css_url='/statics/swagger-ui/swagger-ui.css',
        swagger_favicon_url='/statics/swagger-ui/favicon.png',
    )


def redoc_ui_path(*args, **kwargs):
    return get_redoc_html(*args, **kwargs,
                          redoc_js_url='/statics/swagger-ui/redoc.standalone.js',
                          redoc_favicon_url='/statics/swagger-ui/favicon.png',
                          )


applications.get_swagger_ui_html = swagger_ui_patch
applications.get_redoc_html = redoc_ui_path

app = FastAPI(title="Paddle OCR API", description="基于 Paddle OCR 和 FastAPI 的接口")

# 挂载静态文件
# app.mount('/statics', StaticFiles(directory='statics'), name='statics')
app.mount('/statics', StaticFiles(directory='PaddleOCRFastAPI/statics'), name='statics')

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

app.include_router(ocr.router)
