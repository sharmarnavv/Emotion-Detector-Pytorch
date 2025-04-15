# Facial Emotion Recognition with PyTorch

This repository contains a real-time facial emotion recognition system built with PyTorch. The system can detect faces in webcam video and classify emotions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

![Emotion Detection Demo](https://github.com/username/emotion-detector-app-pytorch/raw/main/demo.gif)

## Features

- Real-time facial emotion detection using webcam
- Pre-trained PyTorch CNN model
- Support for 7 emotion classes
- Detailed Jupyter notebooks for both training and inference
- Comprehensive documentation

## Project Structure

```
emotion-detector-app-pytorch/
├── archive/                  # Dataset directory
│   └── images/               # Emotion image dataset
│       ├── train/            # Training images
│       └── validation/       # Validation images
├── models/                   # Saved model files
│   └── model.pth             # Pre-trained PyTorch model
├── .ipynb_checkpoints/       # Jupyter notebook checkpoints
├── driver.ipynb              # Real-time emotion detection notebook
├── emotion-recognizer.ipynb  # Model training and evaluation notebook
├── .gitignore                # Git ignore file
└── README.md                 # This file
```

## Requirements

- Python 3.6+
- PyTorch
- OpenCV (cv2)
- NumPy
- Matplotlib
- Jupyter Notebook/Lab
- Webcam (for real-time detection)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/username/emotion-detector-app-pytorch.git
   cd emotion-detector-app-pytorch
   ```

2. Install the required packages:
   ```bash
   pip install torch torchvision opencv-python numpy matplotlib jupyter
   ```

3. Download the dataset (if you want to train your own model):
   - The project uses the FER2013 dataset or a similar facial emotion dataset
   - Place the dataset in the `archive/images` directory with `train` and `validation` subdirectories
   - Each emotion should have its own subdirectory (e.g., `train/happy`, `train/sad`, etc.)

## Usage

### Real-time Emotion Detection

To run the real-time emotion detection using your webcam:

1. Open the `driver.ipynb` notebook in Jupyter:
   ```bash
   jupyter notebook driver.ipynb
   ```

2. Run all cells in the notebook
3. The webcam feed will open with emotion detection
4. Press 'q' to exit the detection

### Training Your Own Model

To train your own emotion recognition model:

1. Open the `emotion-recognizer.ipynb` notebook in Jupyter:
   ```bash
   jupyter notebook emotion-recognizer.ipynb
   ```

2. Follow the step-by-step instructions in the notebook:
   - Data loading and preprocessing
   - Model architecture definition
   - Training with class weighting
   - Model evaluation
   - Saving the trained model

## Model Architecture

The emotion recognition model is a Convolutional Neural Network (CNN) with the following architecture:

- Three convolutional blocks, each with:
  - Convolutional layer
  - LeakyReLU activation
  - Batch normalization
  - Max pooling

- Fully connected layers:
  - Flatten layer
  - Dense layer with LeakyReLU activation
  - Dropout for regularization
  - Output layer with 7 units (one for each emotion class)

The model is trained with class weighting to handle the imbalanced dataset.

## Performance

The model achieves approximately 65-70% accuracy on the validation set, which is competitive with state-of-the-art results on facial emotion recognition tasks, given the challenging nature of the problem.

## Limitations

- Emotion detection accuracy can vary depending on lighting conditions
- Some emotions (like disgust and fear) are harder to detect than others
- The model may struggle with certain facial orientations or occlusions

## Future Improvements

- Implement more advanced architectures (e.g., EfficientNet, Vision Transformer)
- Add data augmentation techniques to improve model robustness
- Explore transfer learning from larger pre-trained models
- Implement ensemble methods for better accuracy


