from flask import Flask, request, jsonify
import cv2, numpy as np, base64, os
from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper

app = Flask(__name__)
face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=0)

model_path = os.path.join(os.path.dirname(__file__), 'models', 'inswapper_128.onnx')
swap_model = INSwapper(model_path)

def swap_face(src, tgt):
    sf = face_analyzer.get(src)
    tf = face_analyzer.get(tgt)
    if not sf or not tf:
        return src
    return swap_model.get(src, sf[0], tf[0], paste_back=True)

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    src = cv2.imdecode(np.frombuffer(base64.b64decode(data['source']), np.uint8), cv2.IMREAD_COLOR)
    tgt = cv2.imdecode(np.frombuffer(base64.b64decode(data['target']), np.uint8), cv2.IMREAD_COLOR)
    res = swap_face(src, tgt)
    _, buf = cv2.imencode('.jpg', res)
    return jsonify({'result': base64.b64encode(buf).decode('utf-8')})

@app.route('/')
def index():
    return 'FACEHEX Server Running'
