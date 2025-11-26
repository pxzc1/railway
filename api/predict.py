import json
from io import BytesIO
from PIL import Image
import onnxruntime as ort

MODEL_ONNX = "model.onnx"
CLASS_JSON = "class_to_idx.json"

try:
    with open(CLASS_JSON, "r") as f:
        data = json.load(f)
        class_to_idx = data.get("class_to_idx", data)
except Exception as e:
    print(f"Error loading class_to_idx.json: {e}")
    class_to_idx = {}

idx_to_class = {v: k for k, v in class_to_idx.items()}

try:
    session = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])
    print("ONNX model loaded successfully")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    session = None
    
def preprocess(img: Image.Image):
    img = img.resize((224, 224))
    img = img.convert("RGB")
    pixels = list(img.getdata())
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    chw = [[], [], []]
    for r, g, b in pixels:
        chw[0].append((r / 255.0 - mean[0]) / std[0])
        chw[1].append((g / 255.0 - mean[1]) / std[1])
        chw[2].append((b / 255.0 - mean[2]) / std[2])
    return [chw]

def softmax(logits):
    max_logit = max(logits)
    exps = [pow(2.718281828459045, l - max_logit) for l in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def predict_image(image_bytes: bytes):
    if session is None:
        return {"success": False, "error": "Model not loaded"}

    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = preprocess(img)
        inputs = {session.get_inputs()[0].name: tensor}
        outputs = session.run(None, inputs)[0][0]

        probs = softmax(list(outputs))
        pred_idx = probs.index(max(probs))
        conf = probs[pred_idx] * 100.0
        pred_class = idx_to_class.get(pred_idx, "Unknown")

        return {"success": True, "prediction": pred_class, "confidence": conf}

    except Exception as e:
        return {"success": False, "error": str(e)}
