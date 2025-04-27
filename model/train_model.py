import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shap
import lime
from lime import lime_image
from tensorflow.keras.models import Model
import matplotlib.cm as cm
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
import shutil
import subprocess

class LeafDiseaseClassifier:
    def __init__(self, img_height=224, img_width=224):
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.class_names = None
        
    def _build_model(self, num_classes):
        # Use a pre-trained model for better feature extraction
        base_model = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_height, self.img_width, 3)
        )
        
        # Make base model trainable
        base_model.trainable = True
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def load_and_preprocess_data(self, data_dir):
        images = []
        labels = []
        self.class_names = sorted(os.listdir(data_dir))
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            print(f"Loading {class_name}...")
            for img_name in os.listdir(class_dir):
                try:
                    img_path = os.path.join(class_dir, img_name)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.img_height, self.img_width))
                    img = img / 255.0
                    images.append(img)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {str(e)}")
        
        return np.array(images), np.array(labels)
    
    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        num_classes = len(np.unique(y))
        self.model = self._build_model(num_classes)
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        return self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks
        )
    
    def explain_with_shap(self, image, background_data):
        """Generate SHAP explanation."""
        explainer = shap.DeepExplainer(self.model, background_data)
        shap_values = explainer.shap_values(np.expand_dims(image, axis=0))
        
        plt.figure(figsize=(10, 10))
        shap.image_plot(shap_values, -image, show=False)
        plt.savefig('static/shap_explanation.png', bbox_inches='tight')
        plt.close()
        
        return 'shap_explanation.png'
    
    def explain_with_lime(self, image, top_labels=5):
        """Generate LIME explanation."""
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image, 
            self.model.predict, 
            top_labels=top_labels, 
            hide_color=0, 
            num_samples=1000
        )
        
        # Get the top predicted label
        top_label = explanation.top_labels[0]
        
        plt.figure(figsize=(10, 10))
        temp, mask = explanation.get_image_and_mask(
            top_label, 
            positive_only=True, 
            num_features=10, 
            hide_rest=False
        )
        plt.imshow(mark_boundaries(temp, mask))
        plt.title(f'LIME Explanation (Class: {self.class_names[top_label]})')
        plt.savefig('static/lime_explanation.png', bbox_inches='tight')
        plt.close()
        
        return 'lime_explanation.png'
    
    def explain_with_gradcam(self, image):
        """Generate Grad-CAM explanation."""
        # Get the last conv layer
        last_conv_layer = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break
        
        if last_conv_layer is None:
            last_conv_layer = self.model.layers[-3].name
        
        # Create Grad-CAM model
        grad_model = Model(
            [self.model.inputs],
            [self.model.get_layer(last_conv_layer).output, self.model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(np.expand_dims(image, axis=0))
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_output = conv_output[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Create superimposed visualization
        heatmap = cv2.resize(heatmap, (self.img_width, self.img_height))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cm.jet(heatmap)[..., :3]
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.imshow(heatmap, alpha=0.4)
        plt.title(f'Grad-CAM Explanation (Class: {self.class_names[pred_index]})')
        plt.savefig('static/gradcam_explanation.png', bbox_inches='tight')
        plt.close()
        
        return 'gradcam_explanation.png'
    
    def explain_prediction(self, image, background_data):
        """Generate explanations using multiple techniques."""
        explanations = {
            'shap': self.explain_with_shap(image, background_data),
            'lime': self.explain_with_lime(image),
            'gradcam': self.explain_with_gradcam(image)
        }
        return explanations

def main():
    # Check if dataset exists
    data_dir = "data/leaf_diseases"
    if not os.path.exists(data_dir):
        print("Dataset not found. Please run download_dataset.py first.")
        return
    
    # Initialize classifier
    classifier = LeafDiseaseClassifier()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = classifier.load_and_preprocess_data(data_dir)
    
    # Split data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training model...")
    history = classifier.train(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = classifier.model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save model
    print("Saving model...")
    classifier.model.save('model/leaf_disease_model.h5')
    
    # Save class names
    with open('model/class_names.txt', 'w') as f:
        for class_name in classifier.class_names:
            f.write(f"{class_name}\n")
    
    print("Model training completed and saved!")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        # Clean up in case of error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return 