import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.model.api_models import CodingApiResponse, CodingRequestPayload, ChatCompletionApiResponse, ChatCompletionRequestPayload
from app.request_handler import RequestHandlerProvider

_RESPONSE_TYPE = {
    "chat": ChatCompletionApiResponse,
    "code": CodingApiResponse
}

_REQUEST_PAYLOAD = {
    "chat": ChatCompletionRequestPayload,
    "code": CodingRequestPayload
}


def get_completion_router(prefix: str, request_handler_provider: RequestHandlerProvider) -> APIRouter:
    router = APIRouter(
        prefix=f"/{prefix}", tags=[prefix]
    )

    @router.post("/completion")
    async def create_completion(request: Request, request_payload: _REQUEST_PAYLOAD[prefix], request_handler: Annotated[
        request_handler_provider.get_handler, Depends()]) -> _RESPONSE_TYPE[prefix]:
        return await request_handler.handle_request(request, request_payload)

    @router.on_event("startup")
    async def on_startup():
        asyncio.create_task(request_handler_provider.get_handler().process_request_queue())

    return router

