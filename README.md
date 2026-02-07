# AirTouchpad - Hand Gesture MNIST Digit Recognition

An interactive real-time digit recognition system using hand gestures captured via webcam. Write digits in the air and have them recognized using a deep learning model trained on MNIST dataset.

## 🎯 Features

- **Real-time Hand Detection** - Uses MediaPipe for accurate hand landmark detection
- **Air Drawing** - Write digits in the air using your index finger
- **Multi-digit Recognition** - Recognize multiple digits simultaneously with individual confidence scores
- **Smart Preprocessing** - Automatic cropping, padding, and normalization for accurate predictions
- **Connected Component Analysis** - Detects separate digit regions for independent recognition
- **High Accuracy** - Trained CNN model achieving strong performance on MNIST dataset
- **Gesture Controls**
  - Index Finger: Draw digits
  - Two Fingers Up: Add space between digits
  - Thumb + Pinky Together: Clear canvas

## 📋 Requirements

- Python 3.7+
- Webcam
- PyTorch
- OpenCV
- MediaPipe
- NumPy
- SciPy
- Matplotlib

## 🚀 Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd AirTouchpad
   ```

2. **Install dependencies:**
   ```bash
   pip install torch torchvision opencv-python mediapipe numpy scipy matplotlib
   ```

3. **Train the model (if `mnist_pytorch.pth` doesn't exist):**
   ```bash
   python train_model.py
   ```
   This will:
   - Download MNIST dataset automatically
   - Train the CNN model for 5 epochs
   - Display training loss and test accuracy graphs
   - Save the trained model as `mnist_pytorch.pth`

## 📁 Project Structure

```
AirTouchpad/
├── main_app.py              # Main application with hand gesture recognition
├── train_model.py           # Model training script
├── mnist_pytorch.pth        # Trained model weights
├── training_metrics.png     # Loss and accuracy graphs from training
├── README.md               # This file
└── data/
    └── MNIST/
        └── raw/            # MNIST dataset (downloaded automatically)
```

## 🎮 How to Use

### Running the Application

```bash
python main_app.py
```

### Gesture Controls

1. **Draw Digits** - Raise your index finger upright
   - Your finger position is tracked and drawn on screen (green lines)
   - Write clear, well-spaced digits for better recognition

2. **Add Space** - Raise middle and ring fingers together
   - Creates separation between digits
   - Helps the model recognize multiple digits independently
   - Makes output clearer

3. **Clear Canvas** - Bring thumb and pinky finger together (touch)
   - Clears the entire drawing
   - Ready to draw new digits

### Real-time Feedback

- **Green circles**: Current finger position while drawing
- **Green lines**: Strokes as you write
- **Blue bounding boxes**: Detected digit regions
- **Displayed output**: Shows recognized digits with confidence percentages
  - Example: `Recognized: 5(95%) 3(87%) 8(92%)`

## 🧠 Model Architecture

The trained CNN model (`mnist_pytorch.pth`) uses the following architecture:

```
Net(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (dropout1): Dropout(p=0.25)
  (dropout2): Dropout(p=0.5)
  (fc1): Linear(9216, 128)
  (fc2): Linear(128, 10)
)
```

**Key Details:**
- Input: 28×28 grayscale images (MNIST format)
- 2 Convolutional layers with ReLU activation
- Max pooling and dropout for regularization
- 2 Fully connected layers
- Log softmax output for 10 digit classes (0-9)

## 📊 Training

The `train_model.py` script trains the model with:

- **Dataset**: MNIST (60,000 training, 10,000 test images)
- **Batch Size**: 64
- **Epochs**: 5
- **Optimizer**: Adadelta (lr=1.0)
- **Loss Function**: Negative Log Likelihood (NLL)
- **Device**: GPU (CUDA) if available, otherwise CPU

**Output:**
- Training loss per epoch
- Test accuracy per epoch
- `training_metrics.png` - Visualization graphs
- `mnist_pytorch.pth` - Saved model weights

## ⚙️ Configuration

Edit these values in `main_app.py` to customize behavior:

```python
PROB_THRESHOLD = 0.85  # Minimum confidence score to display prediction
```

Edit these values in `train_model.py` to customize training:

```python
BATCH_SIZE = 64        # Training batch size
EPOCHS = 5             # Number of training epochs
```

## 🔍 Preprocessing Pipeline

When predicting, the app performs:

1. **Contour Detection** - Finds all drawn digit regions
2. **Bounding Box Extraction** - Isolates each digit with padding
3. **Square Conversion** - Adds black padding to maintain aspect ratio
4. **Resizing** - Resizes to 28×28 (MNIST standard)
5. **Normalization** - Applies MNIST mean (0.1307) and std (0.3081)
6. **Prediction** - Passes through trained CNN model

## 🐛 Troubleshooting

### Model not found error
```
Error: 'mnist_pytorch.pth' not found. Run train.py first.
```
**Solution:** Run `python train_model.py` to train and save the model.

### Digits not recognized
- Ensure good lighting for accurate hand detection
- Write digits clearly with proper spacing
- Use the space gesture between digits
- Try raising confidence threshold lower in `PROB_THRESHOLD`

### Low accuracy predictions
- Ensure digits are large enough and well-formed
- Avoid overlapping digits
- Write at a moderate speed for clearer strokes

### Camera not working
- Check if webcam is properly connected
- Verify other applications aren't using the camera
- Adjust `min_detection_confidence` in MediaPipe setup

## 📈 Performance

- **FPS**: Real-time processing at 30+ FPS (depends on GPU)
- **Model Accuracy**: ~98% on MNIST test set
- **Recognition Confidence**: Typically 85-99% for well-written digits
- **Latency**: <50ms per digit prediction

## 🔗 Dependencies Overview

| Package | Purpose |
|---------|---------|
| **PyTorch** | Deep learning framework |
| **OpenCV** | Image processing and webcam capture |
| **MediaPipe** | Hand detection and gesture recognition |
| **NumPy** | Numerical computations |
| **SciPy** | Connected component analysis |
| **Matplotlib** | Training visualization |

## 🎓 How It Works

1. **Capture**: Webcam captures video frames
2. **Detect**: MediaPipe detects hand landmarks (21 key points per hand)
3. **Track**: Finger position tracked and drawn on canvas
4. **Recognize**: Drawn digits sent to pre-trained CNN model
5. **Display**: Predictions with confidence scores shown in real-time

## 📝 Notes

- Model is trained specifically for single digit recognition (0-9)
- Best results with clean, well-formed handwriting
- Multiple digits can be written in one session
- Model runs on GPU if available for faster inference

## 🚦 Quick Start Example

```bash
# 1. Install dependencies
pip install torch torchvision opencv-python mediapipe numpy scipy matplotlib

# 2. Train the model (first time only)
python train_model.py

# 3. Run the application
python main_app.py

# 4. Now write digits in the air!
```

## 📱 Future Enhancements

- [ ] Support for uppercase letters (A-Z)
- [ ] Lowercase letters and special characters
- [ ] Custom gesture training
- [ ] Confidence threshold adjustment via gesture
- [ ] Handwriting history/logging
- [ ] Multi-hand support for simultaneous digit writing

## 📄 License

This project uses open-source libraries. Refer to their respective licenses.

## 👤 Author
Prahas Hegde
MSc AI, THWS
Created as an interactive air-based handwriting recognition system.

---

**Enjoy writing digits in the air! ✋✨**
