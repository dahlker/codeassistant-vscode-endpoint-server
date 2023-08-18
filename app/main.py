from pathlib import Path

import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger

from app.Llm import Llm
from app.generators import CodeGenerator, ChatGenerator
from app.logger import configure_logger
from app.request_handler import RequestHandler
from app.request_handler import RequestHandlerProvider
from app.model.api_models import CompletionType
from app.routers.completion import get_completion_router
from app.routers.feedback import get_feedback_router
from app.util import get_config_from_arguments, ApiConfig, ModelConfig


def read_version():
    return (Path(__file__).parent.parent / "VERSION").read_text().strip()


def add_completion_endpoints(model_config: ModelConfig, router: APIRouter):
    llm = Llm(model_config)
    generator_classes = {
        CompletionType.CODE: CodeGenerator, 
        CompletionType.CHAT: ChatGenerator
    }
    for api_type, generator_class in generator_classes.items():
        generator = generator_class(llm)
        request_handler = RequestHandler(generator=generator)
        router.include_router(get_completion_router(api_type, RequestHandlerProvider(request_handler)))


def add_feedback_endpoint(router):
    router.include_router(get_feedback_router())


def build_app(api_config: ApiConfig, model_config: ModelConfig) -> FastAPI:
    async def verify_token(credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())):
        if credentials.scheme != "Bearer" or not credentials.credentials.startswith(api_config.auth_prefix):
            raise HTTPException(status_code=401, detail="Invalid bearer token")

    app: FastAPI = FastAPI(
        title="TNG Internal LLM Server",
        version=read_version(),
        dependencies=[Depends(verify_token)]
    )

    app.add_middleware(CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"])

    router = APIRouter()
    add_completion_endpoints(model_config, router)
    add_feedback_endpoint(router)
    app.include_router(router)

    return app


def main():
    api_config, model_config, server_config = get_config_from_arguments()
    configure_logger(model_config)
    app = build_app(api_config, model_config)
    uvicorn.run(app, **server_config.model_dump())


if __name__ == '__main__':
    main()
