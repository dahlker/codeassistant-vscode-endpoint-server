import asyncio
from pathlib import Path

import uvicorn
from fastapi import FastAPI, APIRouter

from Llm import Llm
from app.dependencies import RequestHandlerProvider
from app.routers.completion import get_completion_router
from generators import CodeGenerator, ChatGenerator
from request_handler import RequestHandler
from util import get_parser, logger


def read_version():
    return (Path(__file__).parent.parent / "VERSION").read_text().strip()


def get_args():
    args = get_parser().parse_args()

    ssl_certificate, ssl_key = None, None
    if args.ssl_certificate and args.ssl_keyfile:
        ssl_certificate = args.ssl_certificate
        ssl_key = args.ssl_keyfile
    model_name = args.pretrained
    return args.api_type, args.auth_prefix, args.bit_precission, args.dry_run, model_name, ssl_certificate, ssl_key, args.host, args.port


def build_app(api_type: str, request_handler) -> FastAPI:
    app: FastAPI = FastAPI(
        title="TNG Internal Starcoder",
        version=read_version()
    )

    router = APIRouter(
        prefix="/api/v1"
    )
    router.include_router(get_completion_router(api_type, RequestHandlerProvider(request_handler)))
    app.include_router(router)

    @app.on_event("startup")
    async def on_startup():
        asyncio.create_task(request_handler.process_request_queue())

    return app


def create_request_handler(api_type: str, auth_prefix: str, bitsize: int, do_not_load_llm: bool,
                           model_name: str) -> RequestHandler:
    llm = Llm(model_name=model_name, bitsize=bitsize, do_not_load_llm=do_not_load_llm)
    if api_type == 'code':
        generator = CodeGenerator(llm=llm)
    elif api_type == 'chat':
        generator = ChatGenerator(llm=llm)
    else:
        logger.error(f"api_type {api_type} not supported. Use 'code' or 'chat'")
        exit()

    return RequestHandler(generator=generator, auth_prefix=auth_prefix)


def main():
    api_type, auth_prefix, bitsize, do_not_load_llm, model_name, ssl_certificate, ssl_key, host, port = get_args()
    request_handler = create_request_handler(api_type, auth_prefix, bitsize, do_not_load_llm, model_name)
    app = build_app(api_type, request_handler)
    uvicorn.run(app, host=host, port=port, ssl_keyfile=ssl_key, ssl_certfile=ssl_certificate)


if __name__ == '__main__':
    main()
