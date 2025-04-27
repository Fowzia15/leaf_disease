from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from model.train_model import LeafDiseaseClassifier
import shap

app = Flask(__name__)

# Load the trained model
model = None
class_names = []
classifier = None
background_data = None

def load_model():
    global model, class_names, classifier, background_data
    
    try:
        # Initialize classifier
        classifier = LeafDiseaseClassifier()
        
        # Load the saved model
        if not os.path.exists('model/leaf_disease_model.h5'):
            raise FileNotFoundError("Model file not found. Please train the model first.")
            
        model = tf.keras.models.load_model('model/leaf_disease_model.h5')
        classifier.model = model
        
        # Load class names
        if not os.path.exists('model/class_names.txt'):
            raise FileNotFoundError("Class names file not found.")
            
        with open('model/class_names.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        
        # Load some background data for SHAP
        data_dir = "data/leaf_diseases"
        if os.path.exists(data_dir):
            images = []
            for class_name in os.listdir(data_dir)[:1]:  # Just use first class for background
                class_dir = os.path.join(data_dir, class_name)
                for img_name in os.listdir(class_dir)[:10]:  # Use 10 images
                    img_path = os.path.join(class_dir, img_name)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Could not read image: {img_path}")
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    img = img / 255.0
                    images.append(img)
            if images:
                background_data = np.array(images)
            else:
                print("Warning: No valid background images found for SHAP")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
            
        file = request.files['file']
        if not file:
            return jsonify({'error': 'Empty file uploaded'})
            
        # Read and preprocess the image
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Could not read image file'})
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        
        # Make prediction
        pred = model.predict(np.expand_dims(img, axis=0))
        predicted_class = class_names[np.argmax(pred[0])]
        confidence = float(np.max(pred[0]))
        
        # Generate explanations
        explanations = None
        if background_data is not None:
            try:
                explanations = classifier.explain_prediction(img, background_data)
            except Exception as e:
                print(f"Error generating explanations: {str(e)}")
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'explanations': explanations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    try:
        load_model()
        app.run(debug=True)
    except Exception as e:
        print(f"Failed to start application: {str(e)}") 