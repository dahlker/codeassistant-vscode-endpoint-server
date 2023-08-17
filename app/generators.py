import time
import traceback
from typing import List
from uuid import uuid4

from app.Llm import Llm
from app.logger import logger
from app.model.api_models import ChatCompletionRequestPayload, ChatCompletionApiResponse, ChatCompletionApiChoice, ChatMessage, ApiUsage
from app.model.api_models import CodingApiResponse, CodingRequestPayload, CodingParameters
from app.model.api_models import GeneratorBase, GeneratorException


class ChatGenerator(GeneratorBase):
    def __init__(self, llm: Llm = None):
        self.llm = llm
        self.message_prefix = '### '
        if "vicuna" in self.llm.model_name:
            # the LLama tokenizer splits "\n###" into "\n", "##", "#". So we check for "##"
            self.llm.add_stopwords(['\n##'])
        else:
            self.llm.add_stopwords([self.message_prefix.strip()])

    @classmethod
    def generate_default_api_response(cls, message: str, status: int) -> ChatCompletionApiResponse:
        response = ChatCompletionApiResponse(id=status, created=int(time.time()), model=message, choices=[], usage=cls.generate_api_usage(0, 0))
        return response

    @staticmethod
    def generate_api_usage(prompt_tokens: int, completion_tokens: int) -> ApiUsage:
        return ApiUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens)

    def chat_messages_to_prompt(self, chat_messages: List[ChatMessage]) -> str:
        chat_message_strings = [f"{self.message_prefix}{msg.role}: {msg.content}" for msg in chat_messages]
        chat_message_strings.append(f"{self.message_prefix}assistant:")
        return "\n".join(chat_message_strings)

    def generate_api_response(self, answer: str, api_usage: ApiUsage) -> ChatCompletionApiResponse:
        idx = f"chatcmpl-{uuid4()}"
        created = int(time.time())
        chat_message = ChatMessage(role="assistant", content=answer)
        chat_completion = ChatCompletionApiChoice(index=0, message=chat_message, finish_reason="stop")
        choices = [chat_completion]
        return ChatCompletionApiResponse(id=idx, created=created, model=self.llm.model_name, choices=choices, usage=api_usage)

    def get_generation_config(self, request_payload: ChatCompletionRequestPayload) -> dict:
        config_keys = {'temperature': 'temperature', 'top_p': 'top_p', 'max_tokens': 'max_new_tokens'}
        generation_config_dict: dict = {new_k: getattr(request_payload, k) for k, new_k in config_keys.items()}
        return generation_config_dict

    async def generate(self, request_payload: ChatCompletionRequestPayload) -> ChatCompletionApiResponse:
        try:
            prompt = self.chat_messages_to_prompt(request_payload.messages)
            answer, prompt_tokens, completion_tokens = self.llm.generate(prompt, self.get_generation_config(request_payload),
                                                                         remove_prompt_from_reply=True)
            answer = answer.lstrip()
            api_usage: ApiUsage = self.generate_api_usage(prompt_tokens, completion_tokens)
        except (RuntimeError, AttributeError) as e:
            logger.error(f"Llm chat inference error: {str(e)}")
            logger.debug(f"Full stacktrace: \n{traceback.format_exc()}")
            raise GeneratorException("Internal error invoking the model. Please let us know that you are experiencing this error.")
        return self.generate_api_response(answer, api_usage)


class CodeGenerator(GeneratorBase):
    def __init__(self, llm: Llm = None):
        self.llm = llm

    @classmethod
    def generate_default_api_response(cls, message: str, status: int) -> CodingApiResponse:
        response = CodingApiResponse(id=f"codecmpl-{uuid4()}", status=status, generated_text=message)
        return response

    def get_generation_config(self, request_payload: CodingRequestPayload) -> tuple:
        coding_parameters = CodingParameters() if request_payload.parameters is None else request_payload.parameters
        parameters = {}
        stopping_criteria_list = None
        for param, value in coding_parameters.model_dump().items():
            if param == 'stop' and value is not None and len(value) > 0:
                stopping_criteria_list = self.llm.get_stopping_criteria_list(value)
            else:
                parameters[param] = value
        return parameters, stopping_criteria_list

    async def generate(self, request_payload: CodingRequestPayload) -> CodingApiResponse:
        generation_config_dict, stopping_criteria_list = self.get_generation_config(request_payload=request_payload)
        try:
            answer, prompt_tokens, completion_tokens = self.llm.generate(request_payload.inputs, generation_config_dict, stopping_criteria_list=stopping_criteria_list,
                                                                         remove_prompt_from_reply=False)
        except (RuntimeError, AttributeError) as e:
            logger.error(f"Llm code inference error: {str(e)}")
            logger.debug(f"Full stacktrace: \n{traceback.format_exc()}")
            raise GeneratorException("Internal error invoking the model. Please let us know that you are experiencing this error.")
        return self.generate_default_api_response(answer, 200)

