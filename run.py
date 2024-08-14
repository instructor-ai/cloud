from typing import List
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


import instructor
import openai

app = FastAPI()
client = instructor.from_openai(openai.OpenAI(), model="gpt-4o-mini")


class Property(BaseModel):
    name: str
    value: str


class User(BaseModel):
    name: str
    age: int
    properties: List[Property]


@app.post("/v1/extract_user", response_model=User)
def extract_user(text: str):
    user = client.chat.completions.create(
        messages=[
            {"role": "user", "content": f"Extract user from `{text}`"},
        ],
        response_model=User,
    )
    return user


@app.post("/v1/extract_user_stream")
def extract_user_stream(text: str):
    user_stream = client.chat.completions.create_partial(
        messages=[
            {"role": "user", "content": f"Extract user from `{text}`"},
        ],
        response_model=User,
    )

    def stream():
        for partial_user in user_stream:
            yield f"data: {partial_user.model_dump_json()}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
