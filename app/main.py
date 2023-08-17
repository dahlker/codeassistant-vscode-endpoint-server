from pathlib import Path

import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger

from app.Llm import Llm
from app.generators import CodeGenerator, ChatGenerator
from app.logger import configure_logger
from app.request_handler import RequestHandler
from app.request_handler import RequestHandlerProvider
from app.routers.completion import get_completion_router
from app.routers.feedback import get_feedback_router
from app.util import get_config_from_arguments, ApiConfig, ModelConfig


def read_version():
    return (Path(__file__).parent.parent / "VERSION").read_text().strip()


def build_app(api_config: ApiConfig, model_config: ModelConfig) -> FastAPI:
    async def verify_token(credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())):
        if credentials.scheme != "Bearer" or not credentials.credentials.startswith(api_config.auth_prefix):
            raise HTTPException(status_code=401, detail="Invalid bearer token")

    app: FastAPI = FastAPI(
        title="TNG Internal Starcoder",
        version=read_version(),
        dependencies=[Depends(verify_token)]
    )

    router = APIRouter()
    add_completion_endpoint(api_config, model_config, router)
    add_feedback_endpoint(router)
    app.include_router(router)

    return app


def add_completion_endpoint(api_config, model_config, router):
    request_handler = create_request_handler(api_config, model_config)
    router.include_router(get_completion_router(api_config.api_type, RequestHandlerProvider(request_handler)))


def create_request_handler(api_config: ApiConfig, model_config: ModelConfig) -> RequestHandler:
    llm = Llm(model_config)
    if api_config.api_type == 'code':
        generator = CodeGenerator(llm=llm)
    elif api_config.api_type == 'chat':
        generator = ChatGenerator(llm=llm)
    else:
        logger.error(f"api_type {api_config.api_type} not supported. Use 'code' or 'chat'")
        exit()

    return RequestHandler(generator=generator, auth_prefix=api_config.auth_prefix)


def add_feedback_endpoint(router):
    router.include_router(get_feedback_router())


def main():
    api_config, model_config, server_config = get_config_from_arguments()
    configure_logger(api_config, model_config)
    app = build_app(api_config, model_config)
    uvicorn.run(app, **server_config.model_dump())


if __name__ == '__main__':
    main()
