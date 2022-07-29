from model.torch_utils import get_prediction

from PIL import Image
from flask import Flask, jsonify, request
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_name = get_prediction(img_bytes)
        return jsonify({ 'class_name': class_name})


if __name__ == '__main__':
    app.run()