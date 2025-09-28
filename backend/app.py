# backend/app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(ROOT, 'models', 'embeddings_model.joblib')

print('Loading model from', MODEL_PATH)
model_data = joblib.load(MODEL_PATH)
embedder = model_data['embedder']
clf = model_data['clf']
mlb = model_data['mlb']

app = Flask(__name__, static_folder=os.path.join(ROOT, 'frontend'), static_url_path='/')
CORS(app)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/generate', methods=['POST'])
def generate():
    payload = request.get_json(force=True)
    text = payload.get('text', '')

    if not text.strip():
        return jsonify({'error': 'No text provided.'}), 400

    # Encode input text
    emb = embedder.encode([text])
    probs = clf.predict_proba(emb)[0]

    # Always return top 2 hashtags regardless of probability
    top_idx = np.argsort(probs)[::-1][:2]
    chosen = [(mlb.classes_[i], probs[i]) for i in top_idx]

    hashtags = ['#' + tag for tag, _ in chosen]
    scores = [float(p) for _, p in chosen]

    return jsonify({'hashtags': hashtags, 'scores': scores})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
