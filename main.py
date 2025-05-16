import os

from fastapi import FastAPI, HTTPException
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pathlib import Path

from local_llm_prompt import ask_model

BASE_DIR = Path(__file__).resolve().parent.parent

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

app = FastAPI()


@app.get("/t2t")
def get_t2t(request: Request, promt: str = None):
    if not promt:
        return HTTPException(status_code=404, detail="u need to enter promt")

    headers = request.headers

    try:
        token = headers["Authorization"].replace("Bearer ", "")
    except (KeyError, AttributeError):
        raise HTTPException(status_code=401, detail="invalid code")

    if token != os.getenv("AUTH_TOKEN"):
        return HTTPException(status_code=401, detail="invalid token")


    return JSONResponse(
        {"answer": ask_model(promt)}
    )



