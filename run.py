from typing import List
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


import instructor
from configs import load_configs
import openai

app = FastAPI()
client = instructor.from_openai(openai.OpenAI())


for config in load_configs():
    OutputModel = config.create_output_model()
    InputModel = config.create_input_model()
    path = config.path

    @app.post(path, response_model=OutputModel)
    def extract_data(input: InputModel):
        return client.chat.completions.create(
            model=config.model,
            messages=config.messages(input),
            response_model=OutputModel,
        )

    @app.post(f"{path}/list")
    def extract_data_list(input: InputModel):
        objs = client.chat.completions.create_iterable(
            model=config.model,
            messages=config.messages(input),
            response_model=OutputModel,
        )
        return [obj for obj in objs]

    @app.post(f"{path}/list/stream")
    def extract_data_list_stream(input: InputModel):
        def stream():
            for obj in client.chat.completions.create_iterable(
                model=config.model,
                messages=config.messages(input),
                response_model=OutputModel,
                stream=True,
            ):
                yield obj

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post(f"{path}/stream")
    def extract_data_stream(input: InputModel):
        user_stream = client.chat.completions.create_partial(
            model=config.model,
            messages=config.messages(input),
            response_model=OutputModel,
        )

        def stream():
            for partial_user in user_stream:
                yield f"data: {partial_user.model_dump_json()}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")
