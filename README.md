# Cat vs Dog Image Classifier 🐱🐶

Deep learning CNN model for binary image classification of cats and dogs using TensorFlow/Keras, designed for Google Colab with Kaggle dataset integration.

## 📋 Description

This project implements a Convolutional Neural Network (CNN) to classify images of cats and dogs. The model is trained on the Dogs vs Cats dataset from Kaggle and includes comprehensive preprocessing, data augmentation, training callbacks, and evaluation metrics. Built specifically for Google Colab environment with seamless Kaggle API integration.

## ✨ Features

- **Automated Dataset Download**: Direct Kaggle dataset integration via API
- **Advanced Data Augmentation**: Rotation, shifting, zooming, and flipping for robust training
- **Deep CNN Architecture**: 3 convolutional blocks with batch normalization and dropout
- **Smart Training**: Early stopping, learning rate reduction, and model checkpointing
- **Comprehensive Evaluation**: Confusion matrix, classification reports, and accuracy plots
- **Custom Image Testing**: Upload your own images for real-time predictions
- **Visual Analytics**: Training curves, sample predictions, and performance metrics

## 🛠️ Technologies Used

- Python 3.x
- TensorFlow 2.x / Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- PIL (Python Imaging Library)
- Google Colab (recommended)
- Kaggle API

## 📦 Installation & Setup

### For Google Colab (Recommended)

1. **Upload Kaggle API Token**:
   - Get your `kaggle.json` from [Kaggle Account Settings](https://www.kaggle.com/account)
   - Run the notebook and upload when prompted

2. **Dataset Auto-Download**:
   ```python
   # The notebook automatically downloads and extracts the dataset
   !kaggle datasets download -d bhavikjikadara/dog-and-cat-classification-dataset
   ```

### For Local Environment

```bash
# Clone repository
git clone https://github.com/yourusername/cat-vs-dog-classifier.git
cd cat-vs-dog-classifier

# Install dependencies
pip install tensorflow numpy matplotlib seaborn scikit-learn pillow kaggle
```

## 🚀 Quick Start

### Step 1: Setup Kaggle API
```python
# Upload your kaggle.json when prompted
# Script automatically configures Kaggle credentials
```

### Step 2: Download Dataset
```python
# Dataset downloads automatically (~800MB)
# Extracts to /content/PetImages/
```

### Step 3: Train Model
```python
# Run all cells sequentially
# Training takes ~15-20 minutes on Colab GPU
history = model.fit(train_generator, epochs=25, validation_data=val_generator)
```

### Step 4: Test on Custom Images
```python
# Upload your own cat/dog images
uploaded = files.upload()
# Get instant predictions with confidence scores
```

## 🏗️ Model Architecture

```
Input: 128×128×3 RGB Images
    ↓
[Conv Block 1]
├── Conv2D(32 filters, 3×3, ReLU)
├── BatchNormalization
├── MaxPooling2D(2×2)
└── Dropout(0.25)
    ↓
[Conv Block 2]
├── Conv2D(64 filters, 3×3, ReLU)
├── BatchNormalization
├── MaxPooling2D(2×2)
└── Dropout(0.25)
    ↓
[Conv Block 3]
├── Conv2D(128 filters, 3×3, ReLU)
├── BatchNormalization
├── MaxPooling2D(2×2)
└── Dropout(0.3)
    ↓
Flatten
    ↓
Dense(256, ReLU)
    ↓
BatchNormalization
    ↓
Dropout(0.5)
    ↓
Output: Dense(1, Sigmoid) → Cat (0) or Dog (1)
```

## ⚙️ Hyperparameters

```python
# Image Processing
img_size = (128, 128)
batch_size = 32

# Data Augmentation
rotation_range = 30
width_shift_range = 0.1
height_shift_range = 0.1
zoom_range = 0.2
horizontal_flip = True

# Training
epochs = 25 (with early stopping)
validation_split = 0.2 (80/20 train/val)
optimizer = 'adam'
loss = 'binary_crossentropy'
```

## 📊 Training Callbacks

### Early Stopping
- Monitors: `val_loss`
- Patience: 5 epochs
- Restores best weights automatically

### Model Checkpoint
- Saves: `best_cnn_model.h5`
- Monitors: `val_accuracy`
- Keeps only the best performing model

### Learning Rate Reduction
- Reduces LR by factor of 0.2
- Triggers after 3 epochs of no improvement
- Minimum LR: 1e-6

## 📈 Expected Results

- **Training Accuracy**: ~95-98%
- **Validation Accuracy**: ~85-90%
- **Test Predictions**: High confidence on clear images
- **Training Time**: ~15-20 minutes (Colab GPU)

### Sample Performance Metrics

```
Classification Report:
              precision    recall  f1-score   support

         Cat       0.88      0.87      0.87      2500
         Dog       0.87      0.88      0.88      2500

    accuracy                           0.88      5000
```

## 📁 Project Structure

```
cat-vs-dog-classifier/
│
├── project.py                 # Main Colab notebook/script
├── README.md                  # Documentation
├── best_cnn_model.h5         # Saved trained model (after training)
├── cats_vs_dogs_cnn.h5       # Final exported model
└── requirements.txt          # Python dependencies
```

## 🔍 Key Features Explained

### Data Augmentation
Prevents overfitting by creating variations:
- **Rotation**: ±30 degrees
- **Shifting**: 10% horizontal/vertical
- **Zooming**: Up to 20%
- **Flipping**: Horizontal only
- **Normalization**: Pixels scaled to [0,1]

### Regularization Techniques
1. **Dropout**: Randomly deactivates neurons (25%-50%)
2. **Batch Normalization**: Stabilizes learning
3. **L2 Regularization**: Implicit via BatchNorm
4. **Early Stopping**: Prevents overfitting

## 🖼️ Using Custom Images

### Upload and Predict
```python
from google.colab import files
uploaded = files.upload()

# Automatic prediction with confidence
# Output: "Dog 🐶 (Confidence: 0.92)" or "Cat 🐱 (Confidence: 0.88)"
```

### Supported Formats
- JPG, JPEG, PNG
- Any size (auto-resized to 128×128)
- RGB or grayscale (converted to RGB)

## 📊 Visualization Features

1. **Training Curves**: Accuracy and loss over epochs
2. **Confusion Matrix**: True vs predicted classifications
3. **Sample Predictions**: Visual grid of 6 random predictions
4. **Image Distribution**: Dataset statistics and insights

## 🐛 Troubleshooting

### Common Issues

**Kaggle API Error**:
```bash
# Ensure kaggle.json has correct permissions
chmod 600 ~/.kaggle/kaggle.json
```

**Out of Memory**:
```python
# Reduce batch size
batch_size = 16  # instead of 32
```

**Low Accuracy**:
- Train for more epochs
- Increase model complexity
- Check for corrupted images in dataset

## 🚀 Future Improvements

- [ ] Transfer learning with pre-trained models (ResNet, VGG16)
- [ ] Multi-class classification (more pet types)
- [ ] Grad-CAM visualization for interpretability
- [ ] Mobile deployment with TensorFlow Lite
- [ ] Web API with Flask/FastAPI
- [ ] Real-time webcam predictions

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Better data preprocessing pipelines
- Hyperparameter tuning scripts
- Additional evaluation metrics
- Deployment templates

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Dataset**: [Kaggle Dogs vs Cats Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
- **Framework**: TensorFlow/Keras team
- **Platform**: Google Colab for free GPU access
- **Community**: Kaggle and TensorFlow communities

## 📧 Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Pull requests welcome!

## 🎯 Usage Tips

1. **Always use GPU**: Runtime → Change runtime type → GPU (in Colab)
2. **Monitor training**: Watch the validation accuracy to prevent overfitting
3. **Save frequently**: Models are saved automatically, but download them!
4. **Test thoroughly**: Use diverse images to test model robustness

---

⭐ **If this project helps you, please star the repository!**

🐱🐶 **Happy Classifying!**
