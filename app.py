import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
from collections import deque
import time

# CONFIGURATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROB_THRESHOLD = 0.50 
SMOOTHING_FACTOR = 0.2
MIN_CONTOUR_AREA = 400
LOCK_DELAY = 1.0 

# Colors (BGR)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255) # Color for LOCKED state
GRAY = (50, 50, 50)
UI_COLOR = (220, 220, 220)

# MODEL ARCHITECTURE
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

# HELPER CLASSES

class Stabilizer:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.prev_x = None
        self.prev_y = None

    def update(self, x, y):
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
            return x, y
        smooth_x = int(self.alpha * x + (1 - self.alpha) * self.prev_x)
        smooth_y = int(self.alpha * y + (1 - self.alpha) * self.prev_y)
        self.prev_x, self.prev_y = smooth_x, smooth_y
        return smooth_x, smooth_y

    def reset(self):
        self.prev_x = None
        self.prev_y = None

class VirtualButton:
    def __init__(self, pos, text, size=(100, 50), color=UI_COLOR):
        self.pos = pos 
        self.text = text
        self.size = size
        self.color = color
        self.hover_counter = 0
        self.click_threshold = 20

    def draw(self, img):
        x, y = self.pos
        w, h = self.size
        curr_color = (150, 255, 150) if self.hover_counter > 0 else self.color
        cv2.rectangle(img, (x, y), (x+w, y+h), curr_color, -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), BLACK, 2)
        cv2.putText(img, self.text, (x+10, y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 2)
        if self.hover_counter > 0:
            bar_width = int((self.hover_counter / self.click_threshold) * w)
            cv2.rectangle(img, (x, y+h-5), (x+bar_width, y+h), BLUE, -1)

    def is_clicked(self, finger_pos):
        x, y = self.pos
        w, h = self.size
        fx, fy = finger_pos
        if x < fx < x+w and y < fy < y+h:
            self.hover_counter += 1
            if self.hover_counter >= self.click_threshold:
                self.hover_counter = 0
                return True
        else:
            self.hover_counter = 0
        return False

# CORE LOGIC

def detect_and_predict_digits(img, model):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Collect Boxes
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA: continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, w, h])

    # Merge Logic
    merged_boxes = []
    while len(boxes) > 0:
        curr = boxes.pop(0)
        cx, cy, cw, ch = curr
        keep_checking = True
        while keep_checking:
            keep_checking = False
            for i in range(len(boxes) - 1, -1, -1):
                nx, ny, nw, nh = boxes[i]
                if (nx < cx + cw + 20 and nx + nw > cx - 20):
                     new_x = min(cx, nx)
                     new_y = min(cy, ny)
                     new_w = max(cx+cw, nx+nw) - new_x
                     new_h = max(cy+ch, ny+nh) - new_y
                     curr = [new_x, new_y, new_w, new_h]
                     cx, cy, cw, ch = curr
                     boxes.pop(i)
                     keep_checking = True
        merged_boxes.append(curr)

    results = []
    debug_crop = None

    for box in merged_boxes:
        x, y, w, h = box
        pad = 20
        y1 = max(0, y - pad)
        y2 = min(img.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(img.shape[1], x + w + pad)
        digit_crop = img_gray[y1:y2, x1:x2]
        
        h_c, w_c = digit_crop.shape
        if h_c > w_c:
            diff = h_c - w_c
            digit_crop = cv2.copyMakeBorder(digit_crop, 0, 0, diff//2, diff//2, cv2.BORDER_CONSTANT, value=0)
        else:
            diff = w_c - h_c
            digit_crop = cv2.copyMakeBorder(digit_crop, diff//2, diff//2, 0, 0, cv2.BORDER_CONSTANT, value=0)
            
        img_resize = cv2.resize(digit_crop, (28, 28))
        debug_crop = img_resize
        
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
                'bbox': (x, y, w, h),
                'debug_img': img_resize
            })
            
    results.sort(key=lambda k: k['bbox'][0])
    return results

# MAIN LOOP

def main():
    model = Net().to(DEVICE)
    try:
        model.load_state_dict(torch.load('mnist_pytorch.pth', map_location=DEVICE))
        model.eval()
    except FileNotFoundError:
        print("ERROR: 'mnist_pytorch.pth' not found.")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    paintWindow = np.zeros((480, 640, 3))
    bpoints = [deque(maxlen=512)]
    blue_index = 0
    
    stabilizer = Stabilizer(alpha=SMOOTHING_FACTOR)
    
    btn_clear = VirtualButton((530, 15), "CLEAR", size=(90, 45))
    btn_add = VirtualButton((420, 15), "ADD", size=(100, 45))
    btn_reset = VirtualButton((320, 15), "AC", size=(90, 45))
    
    calc_memory = 0
    feedback_text = ""
    feedback_timer = 0
    
    # LOCKING MECHANISM VARIABLES
    last_draw_time = time.time()
    prediction_locked = False
    locked_digit_str = ""
    locked_results = []
    
    print("Air Calculator Started.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        paintWindow = cv2.resize(paintWindow, (w, h))
        
        # Hand Tracking
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        
        is_drawing_now = False # Track if user is actively drawing this frame

        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0].landmark
            idx_x, idx_y = int(landmarks[8].x * w), int(landmarks[8].y * h)
            
            # DRAWING GESTURE
            if landmarks[8].y < landmarks[6].y and landmarks[12].y > landmarks[10].y:
                is_drawing_now = True
                last_draw_time = time.time() # Reset timer
                prediction_locked = False # Unlock while drawing
                
                smooth_x, smooth_y = stabilizer.update(idx_x, idx_y)
                cv2.circle(frame, (smooth_x, smooth_y), 10, GREEN, -1)
                bpoints[blue_index].appendleft((smooth_x, smooth_y))
            else:
                stabilizer.reset()
                bpoints.append(deque(maxlen=512))
                blue_index += 1

            # BUTTONS
            # Clear
            if btn_clear.is_clicked((idx_x, idx_y)):
                paintWindow[:] = 0
                bpoints = [deque(maxlen=512)]
                blue_index = 0
                locked_digit_str = ""
                prediction_locked = False

            # AC
            if btn_reset.is_clicked((idx_x, idx_y)):
                paintWindow[:] = 0
                bpoints = [deque(maxlen=512)]
                blue_index = 0
                calc_memory = 0
                feedback_text = "Cleared"
                feedback_timer = 40
                locked_digit_str = ""
                prediction_locked = False

            # ADD
            if btn_add.is_clicked((idx_x, idx_y)):
                if locked_digit_str != "":
                    try:
                        num_val = int(locked_digit_str)
                        calc_memory += num_val
                        feedback_text = f"Added +{num_val}"
                        feedback_timer = 40
                        
                        # Reset
                        paintWindow[:] = 0
                        bpoints = [deque(maxlen=512)]
                        blue_index = 0
                        locked_digit_str = ""
                        prediction_locked = False
                    except: pass

            mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        # PREDICTION LOCKING LOGIC
        
        # 1. If not drawing and time passed > delay, LOCK prediction
        if not is_drawing_now and (time.time() - last_draw_time > LOCK_DELAY):
            prediction_locked = True
        
        # 2. Run Prediction Only if NOT Locked (Updating)
        if not prediction_locked:
            if np.sum(paintWindow) > 0:
                locked_results = detect_and_predict_digits(paintWindow.astype('uint8'), model)
                locked_digit_str = "".join([str(r['digit']) for r in locked_results])
            else:
                locked_digit_str = ""
                locked_results = []
        
        # DRAW UI
        
        # Draw Ink
        points = [bpoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None: continue
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], GREEN, 8)
                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], WHITE, 35)

        cv2.rectangle(frame, (0,0), (w, 80), GRAY, -1)
        btn_clear.draw(frame)
        btn_add.draw(frame)
        btn_reset.draw(frame)

        # Draw Bounding Boxes
        # BLUE = Updating, YELLOW = Locked (Safe to move hand)
        box_color = YELLOW if prediction_locked else BLUE
        for r in locked_results:
            x, y, w_box, h_box = r['bbox']
            cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), box_color, 2)
            if prediction_locked:
                cv2.putText(frame, "LOCKED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 2)

        # Text Stats
        cv2.putText(frame, f"SUM: {calc_memory}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        status_color = YELLOW if prediction_locked else WHITE
        display_str = locked_digit_str if locked_digit_str else "..."
        cv2.putText(frame, f"Read: {display_str}", (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        if feedback_timer > 0:
            cv2.putText(frame, feedback_text, (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, GREEN, 3)
            feedback_timer -= 1
        
        # Debug View (Bottom Right)
        if locked_results:
            debug_img = locked_results[0]['debug_img']
            debug_view = cv2.resize(debug_img, (100, 100), interpolation=cv2.INTER_NEAREST)
            debug_view_color = cv2.cvtColor(debug_view, cv2.COLOR_GRAY2BGR)
            frame[h-120:h-20, w-120:w-20] = debug_view_color
            cv2.putText(frame, "AI Input", (w-120, h-125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

        cv2.imshow("Air Calculator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()