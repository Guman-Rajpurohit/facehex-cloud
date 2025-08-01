from flask import Flask, request, jsonify
import os, cv2, base64, numpy as np
import gdown
from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper

app = Flask(__name__)

# Model setup
MODEL_PATH = "models/inswapper_128.onnx"
MODEL_URL = "https://drive.google.com/file/d/1mgB6PWtG5JM7wiKAtUGuBwpWqkQBsZ15/view?usp=sharing"  # Replace with your real file ID

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    print("[+] Downloading ONNX model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Init models
face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=0)
swap_model = INSwapper(MODEL_PATH)

def swap_face(frame, target_img):
    src_faces = face_analyzer.get(frame)
    tgt_faces = face_analyzer.get(target_img)
    if not src_faces or not tgt_faces:
        return frame
    return swap_model.get(frame, src_faces[0], tgt_faces[0], paste_back=True)

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.json
        source_b64 = data['source']
        target_b64 = data['target']

        def decode_image(b64):
            img_bytes = base64.b64decode(b64)
            return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        source = decode_image(source_b64)
        target = decode_image(target_b64)
        result = swap_face(source, target)

        _, buf = cv2.imencode('.jpg', result)
        encoded = base64.b64encode(buf).decode('utf-8')
        return jsonify({'result': encoded})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/')
def home():
    return 'FACEHEX CLOUD SERVER ✅'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
