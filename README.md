# Leaf Disease Classifier with Explainable AI

This project implements a Convolutional Neural Network (CNN) for leaf disease classification with explainable AI integration using SHAP (SHapley Additive exPlanations). The web interface allows users to upload leaf images and receive predictions along with visual explanations of the model's decision.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
   - Create a directory structure under `data/leaf_diseases/`
   - Each disease category should have its own subdirectory
   - Example structure:
     ```
     data/
     └── leaf_diseases/
         ├── healthy/
         ├── bacterial_blight/
         ├── leaf_spot/
         └── rust/
     ```

4. Train the model:
```bash
python model/train_model.py
```

5. Run the Flask application:
```bash
python app.py
```

6. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Upload a leaf image using the web interface
2. Click "Predict Disease" to get the classification result
3. View the prediction along with:
   - Disease classification
   - Confidence score
   - SHAP explanation showing which parts of the image influenced the model's decision

## Model Architecture

The CNN model consists of:
- 3 Convolutional layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Dense layers for classification
- Dropout for regularization

## Explainable AI

The project uses SHAP, LIME, and Grad-CAM to provide visual explanations of the model's predictions. The explanations highlight which regions of the leaf image contributed most significantly to the classification decision.

## Requirements

See `requirements.txt` for a complete list of dependencies. 