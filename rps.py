## Kyx setup (GA402RK-L8190ws)
import os

# Force AMD GPU support for 6800S
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
# Fix the Linux windowing crash
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ==========================================
# 1. SETUP
# ==========================================
# Load the trained model
model_path = 'rps_model.keras' # Make sure this file is in the same folder
print(f"Loading model from {model_path}...")
model = load_model(model_path)
print("✅ Model loaded!")

# Define your class names (MUST match the order from training!)
# Standard flow_from_directory sorts alphabetically. 
# Update this list if your order was different.
class_names = ['Paper', 'Rock', 'Scissors'] 

# Initialize Webcam
cap = cv2.VideoCapture(0) # 0 is usually the default webcam

# check if webcam opened
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Starting video stream. Press 'q' to quit.")

# ==========================================
# 2. MAIN LOOP
# ==========================================
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Flip frame so it looks like a mirror
    frame = cv2.flip(frame, 1)

    # --- Define Region of Interest (ROI) ---
    # We will only look at a square box in the center of the screen
    # This helps the model focus on the hand, not the background furniture
    height, width, _ = frame.shape
    box_size = 300
    
    # Calculate coordinates for the square box
    x1 = int(width / 2 - box_size / 2)
    y1 = int(height / 2 - box_size / 2)
    x2 = x1 + box_size
    y2 = y1 + box_size
    
    # Extract the ROI (The part of the image inside the box)
    roi = frame[y1:y2, x1:x2]
    
    # --- Preprocessing ---
    if roi.size != 0: # Check if roi is valid
        # 1. Resize to match model input (224x224)
        roi_resized = cv2.resize(roi, (224, 224))
        
        # 2. Normalize pixel values (0-1) - Same as training!
        roi_normalized = roi_resized.astype('float32') / 255.0
        
        # 3. Add Batch Dimension (1, 224, 224, 3)
        roi_batch = np.expand_dims(roi_normalized, axis=0)

        # --- Prediction ---
        predictions = model.predict(roi_batch, verbose=0)
        predicted_class_idx = np.argmax(predictions)
        confidence = np.max(predictions)
        
        label = class_names[predicted_class_idx]
        
        # --- Visualization ---
        # Draw the box (Green)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display Text
        text = f"{label} ({confidence*100:.1f}%)"
        
        # Color logic: Green if confident, Red if unsure
        color = (0, 255, 0) if confidence > 0.7 else (0, 0, 255)
        
        cv2.putText(frame, text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Show the "Brain" view (what the model sees) in the corner
        # Optional: helps you debug lighting issues
        debug_view = cv2.resize(roi_resized, (100, 100))
        frame[0:100, 0:100] = debug_view

    # Show the final image
    cv2.imshow('Rock Paper Scissors Detector', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()