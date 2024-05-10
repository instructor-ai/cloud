from typing import List, Literal, Optional, Iterable, Type
from pydantic import BaseModel, Field, create_model, field_validator

TypeHint = Literal[
    "str",
    "int",
    "float",
    "bool",
    "str[]",
    "int[]",
    "float[]",
    "bool[]",
]


class Property(BaseModel):
    title: str
    type: TypeHint
    prompt: Optional[str] = None


class OutputSchema(BaseModel):
    name: str
    prompt: Optional[str] = None
    properties: List[Property]


class InputSchema(BaseModel):
    name: str
    properties: List[Property]


class PromptMessage(BaseModel):
    role: str
    content: str


class Config(BaseModel):
    path: str
    output_schema: OutputSchema
    input_schema: InputSchema
    prompt: List[PromptMessage]
    model: str = "gpt-4-turbo"

    @field_validator("path")
    def validate_path(cls, v: str) -> str:
        assert v.startswith("/"), "Path must be absolute"
        return v

    def create_output_model(self) -> Type[BaseModel]:
        types: dict[str, type] = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "str[]": List[str],
            "int[]": List[int],
            "float[]": List[float],
            "bool[]": List[bool],
        }

        return create_model(
            self.output_schema.name,
            **{
                prop.title: (
                    types[prop.type],
                    Field(
                        ...,
                        title=prop.title,
                        description=prop.prompt,
                    ),
                )
                for prop in self.output_schema.properties
            },  # type: ignore
        )  # type: ignore

    def create_input_model(self) -> Type[BaseModel]:
        types: dict[str, type] = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "str[]": List[str],
            "int[]": List[int],
            "float[]": List[float],
            "bool[]": List[bool],
        }

        return create_model(
            self.input_schema.name,
            **{
                prop.title: (
                    types[prop.type],
                    Field(
                        ...,
                        title=prop.title,
                    ),
                )
                for prop in self.input_schema.properties
            },  # type: ignore
        )  # type: ignore

    def messages(self, data: BaseModel) -> List[dict]:
        from jinja2 import Template

        return [
            {
                "role": message.role,
                "content": Template(message.content).render(**data.model_dump()),
            }
            for message in self.prompt
        ]


def load_configs() -> Iterable[Config]:
    import os
    import yaml

    cur_path = os.path.dirname(__file__)

    for root, dirs, files in os.walk(cur_path):
        for filename in files:
            if filename.endswith(".yaml"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r") as f:
                    api_path = file_path.replace(cur_path, "").split(".")[0]

                    content = yaml.safe_load(f)
                    config = Config.model_validate(dict(path=api_path, **content))
                    yield config
