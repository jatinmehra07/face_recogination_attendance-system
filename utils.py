import cv2
import numpy as np
import pickle
import os
import datetime
import time # Used for log attendance timestamp
from collections import deque
import gc # Import garbage collection
import io 

# --- New Imports for Security and Deep Learning ---
try:
    from cryptography.fernet import Fernet
except ImportError:
    print("Warning: 'cryptography' not installed. Install with: pip install cryptography")
    Fernet = None

try:
    import tensorflow as tf
    # NOTE: Using tf.keras.saving.load_model for maximum compatibility
    from tensorflow.keras.models import load_model # Kept for general use
    import h5py # Dependency for Keras HDF5 models
except ImportError:
    print("Warning: 'tensorflow' or 'h5py' not installed. Mask/Liveness models cannot be loaded.")
    tf = None
    load_model = None
    h5py = None
# ----------------------------------------------------

# --- Global Variables ---
mask_detector_model = None
liveness_detector_model = None 
ENCRYPTION_KEY = None
KEY_FILE = 'encrypted_data/secret.key'

# Global history for EAR-based Liveness (deque for frame-by-frame tracking)
BLINK_HISTORY_LEN = 15
landmark_history = deque(maxlen=BLINK_HISTORY_LEN)


# --- 1. ENCODING MANAGEMENT (With Encryption) ---

def generate_and_save_key(key_path=KEY_FILE):
    """Generates a key and saves it to a file."""
    if Fernet is None: return None
    os.makedirs(os.path.dirname(key_path), exist_ok=True)
    key = Fernet.generate_key()
    with open(key_path, "wb") as key_file:
        key_file.write(key)
    return key

def load_encryption_key(key_path=KEY_FILE):
    """Loads or generates the encryption key."""
    global ENCRYPTION_KEY
    if Fernet is None: return None

    if not os.path.exists(key_path):
        print(f"Encryption key not found. Generating new key at {key_path}")
        ENCRYPTION_KEY = generate_and_save_key(key_path)
    else:
        with open(key_path, "rb") as key_file:
            ENCRYPTION_KEY = key_file.read()
    return ENCRYPTION_KEY

# Load the key once upon utility import
load_encryption_key() 

def load_encodings(encodings_file='encrypted_data/face_encodings.pkl'):
    """ Loads known face encodings and names from the encrypted pickle file. """
    global ENCRYPTION_KEY
    if Fernet is None or ENCRYPTION_KEY is None:
        print("Error: Encryption key or Fernet not available. Cannot load encrypted encodings.")
        return [], []

    try:
        if not os.path.exists(encodings_file):
            print("Error: Encodings file not found. Ensure the encoding script was run.")
            return [], []
            
        f = Fernet(ENCRYPTION_KEY)
        with open(encodings_file, 'rb') as ef:
            encrypted_data = ef.read()
            decrypted_data = f.decrypt(encrypted_data)
            known_encodings, known_names = pickle.loads(decrypted_data)
            
        # Convert to numpy array for efficiency in face_recognition
        known_encodings = np.array(known_encodings)
        print(f"Loaded {len(known_encodings)} known faces from encodings file.")
        return known_encodings, known_names
    except Exception as e:
        print(f"Error loading/decrypting encodings: {e}")
        return [], []

def save_encodings(known_encodings, known_names, encodings_file='encrypted_data/face_encodings.pkl'):
    """ Encrypts and saves face encodings to a file. """
    global ENCRYPTION_KEY
    if Fernet is None or ENCRYPTION_KEY is None:
        print("Error: Encryption key or Fernet not available. Cannot save encodings.")
        return False
        
    f = Fernet(ENCRYPTION_KEY)
    
    # 1. Serialize data
    pickled_data = pickle.dumps((known_encodings, known_names))
    
    # 2. Encrypt data
    encrypted_data = f.encrypt(pickled_data)
    
    # 3. Save encrypted data
    os.makedirs(os.path.dirname(encodings_file), exist_ok=True)
    with open(encodings_file, 'wb') as ef:
        ef.write(encrypted_data)
        
    return True


# --- 2. MODEL LOADING (Mask Detector) ---

def load_mask_detector(model_path="models/mask_detector.h5"):
    """ Loads the Masking Detection Model using the final, aggressive Keras loader attempt. """
    global mask_detector_model
    if tf is None: return False

    try:
        if not os.path.exists(model_path):
            print(f"Warning: Mask detection model file not found at {model_path}")
            return False
            
        # FINAL ATTEMPT: Explicitly open in binary read mode ('rb') and pass the file handle
        with open(model_path, "rb") as f:
            # Use the most robust TF/Keras model loading function
            mask_detector_model = tf.keras.saving.load_model(f) 
        
        print("Mask detection model loaded successfully.")
        return True
        
    except Exception as e:
        print(f"Error loading mask model: {e}")
        print("DIAGNOSIS: The persistent 'utf-8' codec error means we cannot read this binary file.")
        print("RECOMMENDATION: The file 'mask_detector.h5' is likely corrupted. Mask checking will be skipped.")
        
        mask_detector_model = False 
        return False

def load_liveness_detector(model_path="models/liveness_detector.h5"):
    """ Placeholder for Liveness Model loading (EAR is used instead). """
    print("Using EAR-based Liveness detection; skipping Liveness Deep Model loading.")
    return True


# --- 3. LIVENESS DETECTION (EAR-based) ---

def eye_aspect_ratio(eye_points):
    """
    eye_points: list of 6 (x, y) points of the eye.
    Calculates the Eye Aspect Ratio (EAR).
    """
    # Calculate the Euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    
    # Calculate the Euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    
    if C == 0: return 0.0 # Avoid division by zero
    
    # Compute the Eye Aspect Ratio
    ear = (A + B) / (2.0 * C)
    return ear

def check_blink_sequence(landmarks_sequence, ear_threshold=0.20):
    """
    Analyzes a deque of recent face landmark dicts for a blink sequence.
    Returns True if blink detected.
    """
    ears = []
    for ls in landmarks_sequence:
        if ls is None:
            ears.append(0.0)
            continue
        
        left_eye = ls.get('left_eye')
        right_eye = ls.get('right_eye')
        
        if left_eye and right_eye:
            ear_l = eye_aspect_ratio(left_eye)
            ear_r = eye_aspect_ratio(right_eye)
            ears.append((ear_l + ear_r) / 2.0)
        else:
            ears.append(0.0)

    # Simple blink detection: a dip below threshold surrounded by higher EAR
    blink_detected = False
    if len(ears) >= 3:
        for i in range(1, len(ears)-1):
            if (ears[i] < ear_threshold and 
                ears[i-1] >= ear_threshold and 
                ears[i+1] >= ear_threshold):
                blink_detected = True
                break
    return blink_detected

def detect_liveness_ear(current_landmarks):
    """
    Uses the global landmark_history to check for a blink.
    """
    global landmark_history
    
    # 1. Update history with current frame's landmarks
    landmark_history.append(current_landmarks)
    
    # 2. Check for a blink in the sequence
    is_live_by_blink = check_blink_sequence(landmark_history, ear_threshold=0.22) # Slightly adjust threshold
    
    # Note: Liveness is confirmed if a blink is found in the recent history.
    return is_live_by_blink, 1.0


# --- 4. MASK DETECTION (Model Inference) ---

def detect_mask(face_image_roi):
    """ Performs mask detection on a cropped and preprocessed face image. """
    global mask_detector_model
    
    # Graceful skip if model loading failed due to corruption
    if mask_detector_model is None or mask_detector_model is False:
        return False, 0.5 

    # face_image_roi MUST be preprocessed (e.g., resized to 224x224, normalized 0-1) in main.ipynb
    
    try:
        # 1. Preprocess: Add batch dimension
        face_input = np.expand_dims(face_image_roi, axis=0) 
        
        # 2. Predict
        prediction = mask_detector_model.predict(face_input, verbose=0)[0]
        
        # ASSUMPTION: Model outputs probability for two classes [Masked, Not Masked]
        # CHECK YOUR MODEL'S OUTPUT ORDER!
        MASK_INDEX = 0 
        
        is_masked = prediction[MASK_INDEX] > 0.5 # Threshold check
        confidence = prediction[MASK_INDEX] if is_masked else 1 - prediction[1 - MASK_INDEX]
        
        return is_masked, confidence.item()
    except Exception as e:
        print(f"Error during mask inference: {e}")
        return False, 0.5 


# --- 5. ATTENDANCE LOGGING ---

def log_attendance(name, status, log_file='attendance/attendance_log.csv'):
    """ Logs the attendance event to a CSV file. """
    dt_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"{dt_string},{name},{status}\n"
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Check if header needs to be written
    write_header = not os.path.exists(log_file)
    
    # Append to the log file
    with open(log_file, 'a') as f:
        if write_header:
            f.write("Timestamp,Name,Status\n")
        f.write(log_entry)
        
    return True


# --- 6. UTILITY FOR DRAWING BOUNDING BOXES (Helper) ---

def draw_info_on_frame(frame, face_location, name, is_live, is_masked):
    """ Draws the bounding box and status information on the frame. """
    top, right, bottom, left = face_location
    
    color = (0, 255, 0) # Default Green
    status_text = name
    
    if name == "Unknown":
        color = (255, 0, 0) # Blue for unknown
    else:
        if not is_live:
            status_text += " | SPOOF! ❌"
            color = (0, 0, 255) # Red for spoof
        elif is_masked:
            # Mask detection is failing, so we'll treat a failed load as 'Mask Check Skipped'
            if mask_detector_model is False:
                 status_text += " | MASK CHECK SKIPPED"
                 color = (255, 165, 0) # Orange
            else:
                 status_text += " | MASKED 😷"
                 color = (255, 165, 0) # Orange for masked
        else:
            status_text += " | PRESENT ✅"
            color = (0, 255, 0) # Green for present

    # Draw box around face
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    
    # Draw label box
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
    
    # Draw text
    cv2.putText(frame, status_text, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    return frame


# --- 7. UTILITY FOR MASK OVERLAY (Data Generation Helper) ---

def overlay_mask_on_face(img, face_landmarks, mask_path):
    """
    Overlays a transparent mask image onto the detected face using landmarks.
    """
    # NOTE: This function requires 'mask.png' (a small image with a transparent background) 
    # to be in the correct path.
    try:
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_img is None:
            print(f"Error: Mask image not found at {mask_path}. Check file path.")
            return img

        nose_bridge = face_landmarks.get('nose_bridge')
        chin = face_landmarks.get('chin')
        
        if not nose_bridge or not chin:
             return img # Skip if key landmarks are missing

        # Calculate mask size and position based on face geometry
        face_width = chin[-1][0] - chin[0][0]
        mask_width = int(face_width * 1.5)
        
        nose_center_x = nose_bridge[2][0]
        nose_center_y = nose_bridge[2][1]

        mask_height = int(mask_img.shape[0] * (mask_width / mask_img.shape[1]))
        resized_mask = cv2.resize(mask_img, (mask_width, mask_height), interpolation=cv2.INTER_AREA)

        # Calculate top-left corner for placement
        x1 = nose_center_x - mask_width // 2
        y1 = nose_center_y - mask_height // 3 

        x2, y2 = x1 + mask_width, y1 + mask_height
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
        
        # Get the ROI on the original image where the mask will be placed
        roi = img[y1:y2, x1:x2]
        
        # Adjust mask size to match the constrained ROI size
        mask_h_adj = roi.shape[0]
        mask_w_adj = roi.shape[1]
        resized_mask = resized_mask[:mask_h_adj, :mask_w_adj] # Crop mask if it was clipped by bounds

        # Alpha blending for transparent overlay (assuming mask has 4 channels)
        if resized_mask.shape[2] == 4:
            alpha_s = resized_mask[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            
            for c in range(0, 3):
                roi[:, :, c] = (alpha_s * resized_mask[:, :, c] +
                                alpha_l * roi[:, :, c])
        else:
             # Simple non-transparent overlay fallback
             roi[:] = resized_mask[:,:,:3] 


        return img
    except Exception as e:
        print(f"Error during mask overlay: {e}")
        return img