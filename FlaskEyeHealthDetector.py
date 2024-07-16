from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import torch
from skimage.morphology import skeletonize
from torchvision import transforms
from PIL import Image
import joblib
import logging
from featureCalculation.feature import calculateVesselMetrics, calculateTortuosity
from models.unet import UNet

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'

logging.basicConfig(level=logging.DEBUG)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def preprocessImage(image_path, transform):
    image = Image.open(image_path).convert("L")
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def predictSingleImage(image_path, model_path, transform, pixel_spacing_mm, decision_tree_path):
    input_image = preprocessImage(image_path, transform)
    unet = UNet(n_channels=1, n_classes=1)
    unet.load_state_dict(torch.load(model_path))
    unet.eval()
    with torch.no_grad():
        output = unet(input_image)
        prediction = torch.sigmoid(output) > 0.5
    prediction_np = prediction.squeeze().cpu().numpy()
    skeleton = skeletonize(prediction_np)
    density, vessel_length_mm = calculateVesselMetrics(prediction_np, pixel_spacing_mm)
    tortuosity = calculateTortuosity(skeleton)
    features = [[density, tortuosity]]
    clf = joblib.load(decision_tree_path)
    prediction_label = clf.predict(features)
    prediction_str = 'DR' if prediction_label == 1 else 'No DR'
    return prediction_str, density, vessel_length_mm, tortuosity

@app.route('/upload', methods=['POST'])
def uploadFile():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            model_path = 'unet_model_best_vein.pth'
            decision_tree_path = 'decision_tree_model.joblib'
            pixel_spacing_mm = 3 / 304
            prediction_str, density, vessel_length_mm, tortuosity = predictSingleImage(
                file_path, model_path, transform, pixel_spacing_mm, decision_tree_path)
            result = {
                'prediction': prediction_str,
                'density': density,
                'vessel_length_mm': vessel_length_mm,
                'tortuosity': tortuosity
            }
            return jsonify(result)
    except Exception as e:
        logging.exception("Error processing the upload")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host="0.0.0.0", port=5000, debug=True)
