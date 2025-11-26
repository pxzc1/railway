from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from api.predict import predict_image

app = FastAPI(title="Flower Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(flower_image: UploadFile = File(...)):
    try:
        image_bytes = await flower_image.read()
        result = predict_image(image_bytes)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}
