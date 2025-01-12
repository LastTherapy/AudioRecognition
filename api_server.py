from fastapi import FastAPI
from pydantic import BaseModel
from llm.openai_mini import improve_recognition
import settings
import uvicorn

app = FastAPI()


class AudioMessage(BaseModel):
    message: str


@app.post("/api/audio_recognition/improve/")
def improve_recognition(data: AudioMessage):
    return improve_recognition(data.message)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.api_port)