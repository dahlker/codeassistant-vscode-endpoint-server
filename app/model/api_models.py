from enum import Enum
from typing import Optional, List

from pydantic import BaseModel


class CompletionType(str, Enum):
    CHAT = "chat"
    CODE = "code"


class CodingParameters(BaseModel):
    max_new_tokens: Optional[int] = 50
    temperature: Optional[float] = 1.0
    do_sample: Optional[bool] = False
    top_p: Optional[float] = 1.0
    stop: Optional[List[str]] = None

    def key(self):
        return (self.max_new_tokens, self.temperature, self.do_sample, self.top_p, tuple(self.stop) if self.stop is not None else None)


class RequestPayload(BaseModel):

    def key(self):
        raise NotImplementedError


class CodingRequestPayload(RequestPayload):
    inputs: str
    parameters: Optional[CodingParameters] = None

    def key(self):
        return self.inputs, self.parameters.key() if self.parameters else ""


class ApiResponse(BaseModel):
    id: str
    cached: bool = False

    def set_is_cached_response(self):
        self.cached = True


class CodingApiResponse(ApiResponse):
    generated_text: str
    status: int


class CompletionRequestPayload(RequestPayload):
    frequence_penalty: Optional[float] = 0.0
    logit_bias: Optional[dict] = None
    max_tokens: Optional[int] = 16
    model: str
    n: Optional[float] = 1
    presence_penalty: Optional[float] = 0.0
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    user: Optional[str] = None

    def key(self) -> tuple:
        return self.model, self.max_tokens, self.temperature, self.user


class TextCompletionRequestPayload(CompletionRequestPayload):
    prompt: str = "<|endoftext|>"
    suffix: Optional[str] = None
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    best_of: Optional[int] = 1

    def key(self):
        return hash((self.model, self.prompt, self.max_tokens, self.temperature, self.user))


class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequestPayload(CompletionRequestPayload):
    messages: List[ChatMessage]

    def key(self):
        return hash((self.model, self.max_tokens, "\n".join([f"{role}{name}: {content}" for role, content, name in self.messages]), self.max_tokens, self.user))


class CompletionApiChoice(BaseModel):
    text: str
    index: int
    logprobs: List[float]
    finish_reason: str


class ApiUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionApiResponse(ApiResponse):
    object: str
    created: int
    model: str
    choices: list
    usage: ApiUsage


class TextCompletionApiResponse(CompletionApiResponse):
    object: str = "text_completion"
    choices: list[CompletionApiChoice]


class ChatCompletionApiChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionApiResponse(CompletionApiResponse):
    object: str = "chat.completion"
    choices: List[ChatCompletionApiChoice]


class GeneratorException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class GeneratorBase:
    async def generate(self, request_payload: BaseModel) -> ApiResponse:
        raise NotImplementedError

    @classmethod
    def generate_default_api_response(cls, message: str, status: int) -> ApiResponse:
        raise NotImplementedError
