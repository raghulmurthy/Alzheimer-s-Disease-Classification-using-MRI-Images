# Alzheimer's Disease Classification using MRI Images

This project implements a Convolutional Neural Network (CNN) to classify different stages of Alzheimer's Disease using MRI scan images. The model achieves 93.36% accuracy on unseen test data.

## Project Overview

The model classifies MRI scans into four categories:
- Non-Demented (Normal)
- Very Mild Demented
- Mild Demented
- Moderate Demented

## Model Architecture

The CNN architecture consists of:
- Multiple Conv2D layers with increasing filters (32 -> 512)
- BatchNormalization after each convolution
- MaxPooling2D layers for dimensionality reduction
- Dropout layers for regularization
- Dense layers for final classification

```python
model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3,3), input_shape=(128,128,3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),
    # ... additional layers
    keras.layers.Dense(4, activation='softmax')
])
```

## Performance Metrics

### Overall Performance
- Accuracy: 93%
- Macro Average: 94%
- Weighted Average: 93%

### Per-Class Performance
- Mild Demented:
  * Precision: 0.95
  * Recall: 0.95
  * F1-score: 0.95

- Moderate Demented:
  * Precision: 1.00
  * Recall: 0.93
  * F1-score: 0.97

- Non-Demented:
  * Precision: 0.92
  * Recall: 0.97
  * F1-score: 0.94

- Very Mild Demented:
  * Precision: 0.95
  * Recall: 0.88
  * F1-score: 0.91

## Dependencies

- TensorFlow
- Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset

The dataset is sourced from Kaggle: "Alzheimer MRI Dataset Classification" which contains MRI scans in parquet format with the following structure:
- Image data (128x128x3)
- Labels indicating disease stage
- Train and test splits

Link to dataset: [Alzheimer MRI Dataset Classification](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset-classification)

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Load and preprocess data:
```python
df = pd.read_parquet('path_to_data.parquet')
```

3. Train model:
```python
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.05)
```

4. Evaluate on test data:
```python
model.evaluate(test_x, test_y)
```

## Model Training Details

- Batch Size: 32
- Epochs: 50
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Early Stopping: Accuracy threshold at 0.9855

## Future Improvements

1. Address class imbalance through:
   - Data augmentation
   - Class weights
   - SMOTE or other oversampling techniques
   
2. Experiment with:
   - Different architectures (ResNet, DenseNet)
   - Cross-validation
   - Ensemble methods