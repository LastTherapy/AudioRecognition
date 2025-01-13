from fastapi import FastAPI, Response
from pydantic import BaseModel
import settings
import uvicorn
from dotenv import load_dotenv


load_dotenv()
from llm.openai_mini import improve_recognition as llm_improve


app = FastAPI()


class AudioMessage(BaseModel):
    message: str


@app.post("/api/audio_recognition/improve/")
def improve_recognition(data: AudioMessage):
    return Response(content=llm_improve(data.message), media_type="text/plain")


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=settings.api_port)