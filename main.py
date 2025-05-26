# main.py
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import ORJSONResponse, StreamingResponse
from dotenv import load_dotenv
from local_llm_prompt import ask_model_async
from vertex_image_gen import generate_image_async
from io import BytesIO

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

AUTH_TOKEN = os.getenv("AUTH_TOKEN")

app = FastAPI(default_response_class=ORJSONResponse)

def verify_token(request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="invalid token")

@app.get("/t2t")
async def get_t2t(request: Request, prompt: str = None, _: None = Depends(verify_token)):
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    answer = await ask_model_async(prompt)
    return {"answer": answer}

@app.get("/t2i")
async def get_t2i(request: Request, prompt: str = None, _: None = Depends(verify_token)):
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    try:
        image = await generate_image_async(prompt)
        img_bytes = BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return StreamingResponse(img_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
