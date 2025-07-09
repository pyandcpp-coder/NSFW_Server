import requests
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from transformers import AutoImageProcessor, AutoModelForImageClassification
try:
    processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
    model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
except Exception as e:
    raise RuntimeError(f"Failed to load Hugging Face model: {e}")

app = FastAPI(
    title="NSFW Image Detection API",
    description="A headless API to detect NSFW content in images. Returns a simple JSON response.",
    version="2.0.0" 
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class SimplePredictionResponse(BaseModel):
    prediction: str # "nsfw" or "sfw"
    confidence: float # Confidence score for the prediction


class ImageRequest(BaseModel):
    image_url: HttpUrl

def process_image_and_predict(image: Image.Image) -> SimplePredictionResponse:
    """
    Processes an image and returns a simplified NSFW/SFW prediction.
    """
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits


    predicted_class_idx = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class_idx]


    probabilities = logits.softmax(-1)[0]
    confidence = probabilities[predicted_class_idx].item()
    
    return SimplePredictionResponse(prediction=label, confidence=confidence)

@app.post("/predict/", response_model=SimplePredictionResponse)
async def predict_nsfw_from_url(request: ImageRequest):
    """
    Accepts an image URL and returns a simple NSFW/SFW prediction.
    """
    try:
        response = requests.get(request.image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image from URL: {e}")
    except Image.UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="The provided URL does not point to a valid image.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

    return process_image_and_predict(image)

@app.post("/predict/upload/", response_model=SimplePredictionResponse)
async def predict_nsfw_from_upload(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file and returns a simple NSFW/SFW prediction.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    if file.size > 10 * 1024 * 1024:  
        raise HTTPException(status_code=400, detail="File size too large. Maximum size is 10MB.")
    
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Image.UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

    return process_image_and_predict(image)

@app.get("/", include_in_schema=False)
def root():
    return {"message": "NSFW Detection API is running"}
    