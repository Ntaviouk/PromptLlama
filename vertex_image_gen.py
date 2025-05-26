import os
from io import BytesIO

from PIL import Image
from dotenv import load_dotenv

from google.cloud import aiplatform
from vertexai.preview.vision_models import ImageGenerationModel
from google.oauth2 import service_account
import pybase64
import json

load_dotenv()

service_account_json = pybase64.b64decode(os.getenv("IMAGEN"))
service_account_info = json.loads(service_account_json)
credentials = service_account.Credentials.from_service_account_info(service_account_info)
project = "imagen-tutorial-461012"
aiplatform.init(project=project, credentials=credentials)


def generate_image(prompt, model_name="imagegeneration@005"):
    try:
        model = ImageGenerationModel.from_pretrained(model_name)
        response = model.generate_images(
            prompt=prompt,
            number_of_images=1,
        )
        image_bytes = response[0]._image_bytes
        image = Image.open(BytesIO(image_bytes))
        return image
    except Exception as e:
        print(f"Error during image generation: {e}")
        raise


def main():
    prompt = input("enter prompt: \n> ")
    img = generate_image(prompt)
    img.show()


if __name__ == "__main__":
    main()
