import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.model.api_models import CodingApiResponse, CodingRequestPayload, ChatCompletionApiResponse, \
    ChatCompletionRequestPayload, CompletionType
from app.request_handler import RequestHandlerProvider

_RESPONSE_TYPE = {
    CompletionType.CHAT: ChatCompletionApiResponse,
    CompletionType.CODE: CodingApiResponse
}

_REQUEST_PAYLOAD = {
    CompletionType.CHAT: ChatCompletionRequestPayload,
    CompletionType.CODE: CodingRequestPayload
}

_API_PREFIX_CONVENTION_TO_MODEL_OPENAI = {
    CompletionType.CHAT: "/v1/chat/completions",
    CompletionType.CODE: "/api/generate"
}


def get_completion_router(prefix: CompletionType, request_handler_provider: RequestHandlerProvider) -> APIRouter:
    router = APIRouter(
        prefix=_API_PREFIX_CONVENTION_TO_MODEL_OPENAI[prefix], tags=[prefix]
    )

    @router.post("/")
    async def create_completion(request: Request, request_payload: _REQUEST_PAYLOAD[prefix], request_handler: Annotated[
        request_handler_provider.get_handler, Depends()]) -> _RESPONSE_TYPE[prefix]:
        return await request_handler.handle_request(request, request_payload)

    @router.on_event("startup")
    async def on_startup():
        asyncio.create_task(request_handler_provider.get_handler().process_request_queue())

    return router
