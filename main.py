from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import json
import onnxruntime
import numpy as np

MODEL_PATH = "model.onnx"
CLASS_TO_IDX_PATH = "class_to_idx.json"

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ort_session = onnxruntime.InferenceSession(MODEL_PATH)

with open(CLASS_TO_IDX_PATH, "r") as f:
    data = json.load(f)
class_to_idx = data["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    img_array = np.transpose(img_array, (2, 0, 1))
    return np.expand_dims(img_array, axis=0).astype(np.float32)
    
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = preprocess_image(image)

        # ONNX inference
        ort_inputs = {ort_session.get_inputs()[0].name: tensor}
        ort_outs = ort_session.run(None, ort_inputs)
        outputs = ort_outs[0][0]

        exp_scores = np.exp(outputs - np.max(outputs))
        probs = exp_scores / exp_scores.sum()

        pred_idx = int(np.argmax(probs))
        pred_class = idx_to_class[pred_idx]
        confidence = float(probs[pred_idx] * 100)

        return JSONResponse({"prediction": pred_class, "confidence": confidence})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
