import os
import base64
import numpy as np
import torch
import torchvision.transforms as transforms
from flask import Flask, request, render_template, jsonify
from flask import send_file
from torchvision.models import vgg16
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import json
from urllib.parse import unquote

app = Flask(__name__, static_url_path='/static', static_folder='search_images')

image_paths = []
model = vgg16(pretrained=True).features
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_map = {}  # Dictionary to store pre-extracted features
cache_file = 'feature_map_cache.json'

def extract_image_features(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0)
    img = img.to(device)
    features = model(img).reshape(-1)
    return features.detach().cpu().numpy().tolist()

def compare_features(features1, features2):
    features1 = np.array(features1).reshape(1, -1)
    features2 = np.array(features2).reshape(1, -1)
    similarity = cosine_similarity(features1, features2)
    return similarity[0][0]

def load_feature_maps(folder_path):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as cache:
            try:
                feature_map.update(json.load(cache))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    else:
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                if file_name.endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(root, file_name)
                    features = extract_image_features(file_path)
                    feature_map[file_path] = features
        save_cache()

def save_cache():
    with open(cache_file, 'w') as cache:
        json.dump(feature_map, cache)

def add_feature_to_cache(file_path):
    features = extract_image_features(file_path)
    feature_map[file_path] = features
    save_cache()

def delete_feature_from_cache(file_path):
    del feature_map[file_path]
    save_cache()

def update_feature_in_cache(file_path):
    delete_feature_from_cache(file_path)
    add_feature_to_cache(file_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare_images():
    # Get the uploaded image file from the request
    uploaded_file = request.files['image']
    
    # Save the image to a temporary file
    temp_image_path = 'temp_image.jpg'
    uploaded_file.save(temp_image_path)

    # Extract features from the uploaded image
    uploaded_features = extract_image_features(temp_image_path)

    # Compare the features against the pre-extracted features
    similarities = {}
    for image_path, features in feature_map.items():
        similarity = compare_features(uploaded_features, features)
        similarities[image_path] = similarity

    # Sort the similarities in descending order
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # Delete the temporary image file
    os.remove(temp_image_path)

    # Render the index.html template with the results
    return render_template('index.html', image_paths=sorted_similarities)

@app.route('/file/<path:filename>')
def static_proxy(filename):
    # Build the full path to the image file
    image_path = os.path.join(filename)

    # Return the image file with the appropriate MIME type
    return send_file(image_path, mimetype='image/jpeg')


if __name__ == '__main__':
    # Provide the path to the folder containing pre-extracted features
    folder_path = 'search_images'
    load_feature_maps(folder_path)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug = True)
