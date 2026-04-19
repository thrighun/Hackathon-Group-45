from fastapi import FastAPI
from pydantic import BaseModel
from inference.inference import JDModel

app = FastAPI()

jd = JDModel()   # 🔥 load once

class InputData(BaseModel):
    text: str
    reset: bool = False


@app.post("/chat")
def chat(data: InputData):

    if data.reset:
        jd.reset()
        return {"message": "reset done"}

    result = jd(data.text)

    return {"json_data": result}