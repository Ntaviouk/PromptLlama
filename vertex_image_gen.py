# vertex_image_gen.py
import os
import json
import pybase64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from google.cloud import aiplatform
from vertexai.preview.vision_models import ImageGenerationModel
from google.oauth2 import service_account
from functools import lru_cache
from starlette.concurrency import run_in_threadpool

load_dotenv()

_project_id = "imagen-tutorial-461012"


@lru_cache()
def get_credentials():
    service_account_json = pybase64.b64decode(os.getenv("IMAGEN"))
    service_account_info = json.loads(service_account_json)
    return service_account.Credentials.from_service_account_info(service_account_info)


@lru_cache()
def get_model(model_name="imagegeneration@005"):
    credentials = get_credentials()
    aiplatform.init(project=_project_id, credentials=credentials)
    return ImageGenerationModel.from_pretrained(model_name)


def generate_image(prompt, model_name="imagegeneration@005"):
    model = get_model(model_name)
    response = model.generate_images(prompt=prompt, number_of_images=1)
    image_bytes = response[0]._image_bytes
    return Image.open(BytesIO(image_bytes))


async def generate_image_async(prompt: str):
    return await run_in_threadpool(generate_image, prompt)
