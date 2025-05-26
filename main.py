import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv
from local_llm_prompt import ask_model
from vertex_image_gen import generate_image
from io import BytesIO

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

app = FastAPI()


def verify_token(request: Request):
    try:
        token = request.headers["Authorization"].replace("Bearer ", "")
    except (KeyError, AttributeError):
        raise HTTPException(status_code=401, detail="invalid code")

    if token != os.getenv("AUTH_TOKEN"):
        raise HTTPException(status_code=401, detail="invalid token")


@app.get("/t2t")
def get_t2t(request: Request, prompt: str = None):
    if not prompt:
        raise HTTPException(status_code=404, detail="u need to enter prompt")

    verify_token(request)

    return JSONResponse(
        {"answer": ask_model(prompt)}
    )


@app.get("/t2i")
def get_t2i(request: Request, prompt: str = None):
    if not prompt:
        raise HTTPException(status_code=404, detail="u need to enter prompt")

    verify_token(request)

    try:
        image = generate_image(prompt)
        img_bytes = BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        return StreamingResponse(img_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
