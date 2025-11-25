import json
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import onnxruntime as ort

# ----------------------
# CONFIG
# ----------------------
MODEL_ONNX = "model.onnx"           # Your ONNX file
CLASS_JSON = "class_to_idx.json"    # Your class mapping

# ----------------------
# Load class mapping
# ----------------------
try:
    with open(CLASS_JSON, "r") as f:
        data = json.load(f)
        class_to_idx = data.get("class_to_idx", data)
except Exception as e:
    print(f"Error loading class_to_idx: {e}")
    class_to_idx = {}

idx_to_class = {v: k for k, v in class_to_idx.items()}

# ----------------------
# Load ONNX model
# ----------------------
try:
    session = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])
    print("ONNX model loaded successfully")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    session = None

# ----------------------
# Preprocessing (pure Python)
# ----------------------
def preprocess(img: Image.Image):
    img = img.resize((224, 224))
    img = img.convert("RGB")
    pixels = list(img.getdata())

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # HWC → CHW
    chw = [[], [], []]
    for r, g, b in pixels:
        chw[0].append((r / 255.0 - mean[0]) / std[0])
        chw[1].append((g / 255.0 - mean[1]) / std[1])
        chw[2].append((b / 255.0 - mean[2]) / std[2])

    # Return 1x3x224x224 as nested list
    return [[chw[0], chw[1], chw[2]]]

# ----------------------
# Softmax (pure Python)
# ----------------------
def softmax(logits):
    max_logit = max(logits)
    exps = [2.718281828459045 ** (l - max_logit) for l in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

# ----------------------
# Prediction
# ----------------------
def predict_image(image_bytes):
    if session is None:
        return {"success": False, "error": "Model failed to load"}

    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = preprocess(img)

        inputs = {session.get_inputs()[0].name: tensor}
        outputs = session.run(None, inputs)[0][0]  # 1D array

        probs = softmax(list(outputs))
        pred_idx = probs.index(max(probs))
        pred_class = idx_to_class.get(pred_idx, "Unknown")
        conf = probs[pred_idx] * 100.0

        return {"success": True, "prediction": pred_class, "confidence": conf}

    except Exception as e:
        return {"success": False, "error": str(e)}

# ----------------------
# FastAPI app
# ----------------------
app = FastAPI()

# Allow CORS so Vercel can call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_image(image_bytes)
    return result