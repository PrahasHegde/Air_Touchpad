# Air MNIST: A Hand Gesture-Based Digit Recognition System Using PyTorch and MediaPipe

import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
from collections import deque
from scipy import ndimage

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROB_THRESHOLD = 0.85 

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

# --- MODEL ARCHITECTURE (Must match train.py) ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.log_softmax(x, dim=1)
        return output

# Load Model
model = Net().to(DEVICE)
try:
    model.load_state_dict(torch.load('mnist_pytorch.pth', map_location=DEVICE))
    model.eval()
except FileNotFoundError:
    print("Error: 'mnist_pytorch.pth' not found. Run train.py first.")
    exit()

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# State Variables
paintWindow = np.zeros((480, 640, 3)) # Black canvas
bpoints = [deque(maxlen=512)]
blue_index = 0
predicted_digits = [] # Store recognized digits with positions
last_drawing_time = 0

def get_prediction(img):
    """
    Smart Preprocessing:
    1. Find Bounding Box of the drawing (Crop only the digit)
    2. Add padding to make it square
    3. Resize to 28x28
    4. Normalize & Predict
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Find the bounding box of the drawn digit
    coords = cv2.findNonZero(img_gray)
    x, y, w, h = cv2.boundingRect(coords)
    
    # Crop the digit with some padding
    pad = 15
    # Ensure crop doesn't go out of bounds
    y1 = max(0, y - pad)
    y2 = min(img.shape[0], y + h + pad)
    x1 = max(0, x - pad)
    x2 = min(img.shape[1], x + w + pad)
    
    digit_crop = img_gray[y1:y2, x1:x2]
    
    # 2. Make it square (add black padding) to match MNIST style
    h_crop, w_crop = digit_crop.shape
    if h_crop > w_crop:
        diff = h_crop - w_crop
        digit_crop = cv2.copyMakeBorder(digit_crop, 0, 0, diff//2, diff//2, cv2.BORDER_CONSTANT, value=0)
    else:
        diff = w_crop - h_crop
        digit_crop = cv2.copyMakeBorder(digit_crop, diff//2, diff//2, 0, 0, cv2.BORDER_CONSTANT, value=0)
        
    # 3. Resize to 28x28
    img_resize = cv2.resize(digit_crop, (28, 28))
    
    # Prepare for PyTorch
    tensor = torch.tensor(img_resize, dtype=torch.float32).to(DEVICE)
    tensor = tensor / 255.0 
    tensor = (tensor - 0.1307) / 0.3081 # Normalize
    tensor = tensor.unsqueeze(0).unsqueeze(0) # (1, 1, 28, 28)
    
    with torch.no_grad():
        output = model(tensor)
        probs = torch.exp(output)
        
    prob_val, class_idx = torch.max(probs, 1)
    return class_idx.item(), prob_val.item(), (x1, y1, x2-x1, y2-y1)

def detect_and_predict_digits(img):
    """
    Detect separate digit regions using connected components
    and predict each one independently
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY)
    
    # Label connected components
    labeled, num_features = ndimage.label(binary)
    
    results = []
    
    # Find contours for each digit
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Filter out noise (very small areas)
        if area < 100:
            continue
        
        # Extract digit region with padding
        pad = 15
        y1 = max(0, y - pad)
        y2 = min(img.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(img.shape[1], x + w + pad)
        
        # Create a copy for this digit
        digit_img = img[y1:y2, x1:x2].copy()
        digit_gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
        
        # Make it square
        h_crop, w_crop = digit_gray.shape
        if h_crop > w_crop:
            diff = h_crop - w_crop
            digit_gray = cv2.copyMakeBorder(digit_gray, 0, 0, diff//2, diff//2, cv2.BORDER_CONSTANT, value=0)
        else:
            diff = w_crop - h_crop
            digit_gray = cv2.copyMakeBorder(digit_gray, diff//2, diff//2, 0, 0, cv2.BORDER_CONSTANT, value=0)
        
        # Resize to 28x28
        img_resize = cv2.resize(digit_gray, (28, 28))
        
        # Predict
        tensor = torch.tensor(img_resize, dtype=torch.float32).to(DEVICE)
        tensor = tensor / 255.0
        tensor = (tensor - 0.1307) / 0.3081
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            output = model(tensor)
            probs = torch.exp(output)
        
        prob_val, class_idx = torch.max(probs, 1)
        
        if prob_val.item() > PROB_THRESHOLD:
            results.append({
                'digit': class_idx.item(),
                'prob': prob_val.item(),
                'bbox': (x, y, w, h)
            })
    
    return results

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)

print("Air MNIST Started. Index Finger to Draw. Two Fingers Up for Space. Thumb+Pinky to Clear.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    paintWindow = cv2.resize(paintWindow, (w, h)) # Sync dimensions
    
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    
    pred_text = ""

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0].landmark
        
        idx_x, idx_y = int(landmarks[8].x * w), int(landmarks[8].y * h)
        thumb_x, thumb_y = int(landmarks[4].x * w), int(landmarks[4].y * h)
        pinky_x, pinky_y = int(landmarks[20].x * w), int(landmarks[20].y * h)
        middle_x, middle_y = int(landmarks[12].x * w), int(landmarks[12].y * h)
        
        # CLEAR GESTURE (Thumb touches Pinky)
        dist_clear = np.hypot(thumb_x - pinky_x, thumb_y - pinky_y)
        if dist_clear < 30:
            paintWindow[:] = 0
            bpoints = [deque(maxlen=512)]
            blue_index = 0
            predicted_digits = []
            cv2.putText(frame, "CLEARED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
            
        # SPACE GESTURE (Middle & Ring finger touch / two fingers up)
        elif landmarks[12].y < landmarks[10].y and landmarks[16].y < landmarks[14].y:
            # Two fingers up = space, finalize current digit
            if len(bpoints[blue_index]) > 5:
                last_drawing_time = 0  # Trigger prediction
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            cv2.putText(frame, "SPACE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, BLUE, 2)
            
        # DRAWING GESTURE (Index Finger Up)
        elif landmarks[8].y < landmarks[6].y:
            cv2.circle(frame, (idx_x, idx_y), 10, GREEN, -1)
            bpoints[blue_index].appendleft((idx_x, idx_y))
            last_drawing_time = cv2.getTickCount()
            
        # HOVER
        else:
             if len(bpoints[blue_index]) > 0:
                 bpoints.append(deque(maxlen=512))
                 blue_index += 1

        mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    # Draw lines
    points = [bpoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], GREEN, 10)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], WHITE, 25)

    # Predict - Use new multi-digit detection
    if np.sum(paintWindow) > 0:
        try:
            results = detect_and_predict_digits(paintWindow.astype('uint8'))
            
            # Draw bounding boxes and predictions
            y_offset = 40
            pred_text = "Recognized: "
            for result_item in results:
                digit = result_item['digit']
                prob = result_item['prob']
                x, y, w_box, h_box = result_item['bbox']
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), BLUE, 2)
                
                # Add to prediction text
                pred_text += f"{digit}({prob*100:.0f}%) "
                
        except Exception as e:
            pass  # Handle errors silently

    cv2.putText(frame, pred_text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2)
    
    cv2.imshow("Air MNIST", frame)
    # cv2.imshow("Canvas", paintWindow.astype('uint8')) # Optional debug view

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()