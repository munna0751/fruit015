from flask import Flask, request, jsonify
import torch
from PIL import Image
import io

app = Flask(__name__)

# Load model (adjust path if needed)
model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model='weights/best.pt', trust_repo=True)
model.eval()

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert("RGB")

    results = model(img)
    detections = results.pandas().xyxy[0].to_dict(orient="records")

    output = []
    for det in detections:
        output.append({
            "class": det["name"],
            "confidence": round(float(det["confidence"]), 3),
            "bbox": [int(det["xmin"]), int(det["ymin"]), int(det["xmax"]), int(det["ymax"])]
        })

    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
