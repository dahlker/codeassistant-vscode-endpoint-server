import argparse
import logging

from pydantic import BaseModel, Field

from app.model.api_models import CompletionType

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('app')


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--pretrained', type=str, default='starcoder')
    parser.add_argument('--api-type', type=CompletionType, default=CompletionType.CODE)
    parser.add_argument('--bit-precision', type=int, default=16)
    parser.add_argument('--auth-prefix', type=str, default='<secret_key>')
    parser.add_argument('--ssl-certificate', type=str)
    parser.add_argument('--ssl-keyfile', type=str)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--device', type=str, default="")
    return parser


class ServerConfig(BaseModel):
    host: str
    port: int
    ssl_keyfile: str | None
    ssl_certfile: str | None

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        ssl_certificate, ssl_key = None, None
        if args.ssl_certificate and args.ssl_keyfile:
            ssl_certificate = args.ssl_certificate
            ssl_key = args.ssl_keyfile
        return cls.parse_obj(vars(args) | {"ssl_keyfile": ssl_key, "ssl_certfile": ssl_certificate})


class ConfigModel(BaseModel):
    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls.parse_obj(vars(args))


class ApiConfig(ConfigModel):
    api_type: CompletionType
    auth_prefix: str


class ModelConfig(ConfigModel):
    model_name: str = Field(alias="pretrained")
    bitsize: int = Field(alias="bit_precision")
    do_not_load_llm: bool = Field(alias="dry_run", default=False)
    device: str | None = None


def get_config_from_arguments() -> tuple[ApiConfig, ModelConfig, ServerConfig]:
    args = get_parser().parse_args()
    return ApiConfig.from_args(args), ModelConfig.from_args(args), ServerConfig.from_args(args)
