import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image
import os
import glob
import time
import pyperclip
import zipfile

# ===========================
# Install gdown if not already installed
# ===========================
try:
    import gdown
except ImportError:
    os.system('pip install gdown')
    import gdown

# ===========================
# Helper Functions
# ===========================
def download_file_from_drive(drive_url, output_path):
    """Download a file from Google Drive given a sharable URL."""
    file_id = None
    if "id=" in drive_url:
        file_id = drive_url.split("id=")[-1]
    elif "file/d/" in drive_url:
        file_id = drive_url.split("file/d/")[1].split("/")[0]

    if file_id:
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
    else:
        st.error("Invalid Google Drive URL!")

def download_and_unzip(drive_url, extract_to):
    """Download ZIP file from Drive and unzip it."""
    zip_path = "temp.zip"
    download_file_from_drive(drive_url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)

# ===========================
# Load Custom CSS for Styling
# ===========================
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Skipping custom styling.")

local_css("style.css")

st.markdown("""
    <style>
        textarea {
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

# ===========================
# Initialize MediaPipe
# ===========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ===========================
# Load Trained Models
# ===========================
@st.cache_resource
def load_all_models():
    l_model_path = "L_model.h5"
    n_model_path = "N_model.h5"
    w_model_path = "W_model.h5"

    # Paste your Google Drive model links here
    l_model_url = "https://drive.google.com/file/d/1oU978RNbe_MJFK9YQGUqBj1R_0Efkzvf/view?usp=sharing"
    n_model_url = "https://drive.google.com/file/d/1beVVBz21qQpSmA2GOsj8Sx-8Gx1MzsW2/view?usp=sharing"
    w_model_url = "https://drive.google.com/file/d/1oS2uZQzCElx7s9rP0oYrPSmgEdrECqMr/view?usp=sharing"

    if not os.path.exists(l_model_path):
        download_file_from_drive(l_model_url, l_model_path)
    if not os.path.exists(n_model_path):
        download_file_from_drive(n_model_url, n_model_path)
    if not os.path.exists(w_model_path):
        download_file_from_drive(w_model_url, w_model_path)

    letter_model = load_model(l_model_path)
    number_model = load_model(n_model_path)
    word_model = load_model(w_model_path)

    return letter_model, number_model, word_model

letter_model, number_model, word_model = load_all_models()

# ===========================
# Labels
# ===========================
letter_labels = {i: chr(65+i) for i in range(26)}  # A-Z
number_labels = {i: str(i) for i in range(10)}     # 0-9
word_labels = {
    0: "afraid", 1: "agree", 2: "assistance", 3: "bad", 4: "become", 5: "college",
    6: "doctor", 7: "from", 8: "pain", 9: "pray", 10: "secondary", 11: "skin",
    12: "small", 13: "specific", 14: "stand", 15: "today", 16: "warn", 17: "which",
    18: "work", 19: "you", 20: "are", 21: "is", 22: "do"
}

# ===========================
# Preprocessing Functions
# ===========================
def extract_keypoints_from_image(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image_np = np.array(image)
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        elif len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        elif len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

        results = hands.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            return np.array(keypoints).reshape(1, 1, 63).astype(np.float32)
    return None

# ===========================
# Streamlit UI Setup
# ===========================
st.sidebar.image("apac_logo.jpg", width=120)
st.sidebar.title("APAC")

st.title("Introducing APAC – AI-Powered Accessibility Chatbot")
st.write(
    "APAC is a chatbot that responds to sign language, using AI to convert sign language gestures "
    "to text and vice versa. Whether you use sign language, text, or commands, APAC ensures seamless interaction."
)

# ===========================
# Sidebar Input Selection
# ===========================
input_choice = st.selectbox("Select Input Type", ("Upload Image", "Live Webcam", "Text to Sign"))
if "sentence" not in st.session_state:
    st.session_state.sentence = ""

# ===========================
# Upload Image Handling
# ===========================
if input_choice == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        keypoints = extract_keypoints_from_image(image)

        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized)

        if img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif img_array.shape[2] == 1:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

        img_array = img_array / 255.0
        img_input = np.expand_dims(img_array, axis=0)

        if keypoints is not None:
            keypoints_input = np.array(keypoints).reshape(1, -1)

            pred_letter = letter_model.predict(img_input)
            pred_number = number_model.predict(img_input)

            conf_letter = np.max(pred_letter)
            conf_number = np.max(pred_number)

            label_letter = letter_labels.get(np.argmax(pred_letter), "Unknown")
            label_number = number_labels.get(np.argmax(pred_number), "Unknown")

            confidences = [conf_letter, conf_number]
            labels = [label_letter, label_number]
            best_index = np.argmax(confidences)
            best_label = labels[best_index]
            best_confidence = confidences[best_index]

            predicted_label = best_label if best_confidence >= 0.7 else "Uncertain"
            st.text_area("Translated Output", value=predicted_label, height=150, key="output_box", disabled=True)

            if st.button("Copy Text", key="copy_text_upload"):
                pyperclip.copy(predicted_label)
                st.success("Text copied to clipboard!")
        else:
            st.warning("No hand detected or unable to extract keypoints.")

# ===========================
# Live Webcam Handling
# ===========================
if input_choice == "Live Webcam":
    if st.button("Start Webcam for Live Translation", key="start_webcam"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        sentence = ""

        if "stop_webcam" not in st.session_state:
            st.session_state.stop_webcam = False

        if st.button("Stop Webcam", key="stop"):
            st.session_state.stop_webcam = True

        if st.button("Reset", key="reset"):
            sentence = ""
            st.session_state.sentence = ""

        st.info("Webcam started! Show your signs.")

        if 'counter' not in st.session_state:
            st.session_state.counter = 0

        last_prediction_time = time.time()
        previous_word = None
        sentence_output = st.empty()

        with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or st.session_state.stop_webcam:
                    st.info("Webcam Stopped.")
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                current_time = time.time()

                if results.multi_hand_landmarks:
                    detected_labels = []

                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        if current_time - last_prediction_time >= 4:
                            resized_frame = cv2.resize(frame, (224, 224))
                            normalized_frame = resized_frame / 255.0
                            image_input = np.expand_dims(normalized_frame, axis=0)

                            pred_word = word_model.predict(image_input)
                            conf_word = np.max(pred_word)
                            label_word = word_labels.get(np.argmax(pred_word), "Unknown")

                            if conf_word >= 0.7:
                                detected_labels.append(label_word)

                    if detected_labels:
                        final_prediction = " ".join(set(detected_labels))
                        if final_prediction != previous_word:
                            sentence += " " + final_prediction
                            previous_word = final_prediction
                            last_prediction_time = current_time

                stframe.image(frame, channels="BGR")
                sentence_output.text_area(f"Translated Output {st.session_state.counter}", value=sentence.strip(),
                                          height=150, key=f"sentence_output_{st.session_state.counter}", disabled=True)

                st.session_state.counter += 1
                st.session_state.sentence = sentence.strip()

        cap.release()
        cv2.destroyAllWindows()

    st.text_area("Final Translated Output", value=st.session_state.sentence,
                 height=150, key="final_sentence_output", disabled=True)

    if st.button("Copy Text", key="copy_text_live"):
        if st.session_state.sentence.strip():
            pyperclip.copy(st.session_state.sentence.strip())
            st.success("Text copied to clipboard!")
        else:
            st.warning("No text to copy!")

# ===========================
# Text to Sign Handling
# ===========================
if input_choice == "Text to Sign":
    text_input = st.text_input("Enter text to translate to sign language:")

    letters_dir = "letters"
    numbers_dir = "numbers"

    # Paste your ZIP links here
    letters_zip_url = "https://drive.google.com/file/d/1KBzm4W4RoCRBl3ibk28FOvBeHzVoa4U_/view?usp=sharing"
    numbers_zip_url = "https://drive.google.com/file/d/1dXaMoQS-i31q-pfQQQKvceP3znq5F6CP/view?usp=sharing"

    if not os.path.exists(letters_dir):
        download_and_unzip(letters_zip_url, letters_dir)
    if not os.path.exists(numbers_dir):
        download_and_unzip(numbers_zip_url, numbers_dir)

    if text_input:
        words = text_input.split()
        st.write("Corresponding Sign Language Images:")

        for word in words:
            word_lower = word.lower()

            if len(word_lower) == 1 and word_lower.isalpha():
                folder_path = os.path.join(letters_dir, word_lower)
            elif word_lower.isdigit():
                folder_path = os.path.join(numbers_dir, word_lower)
            else:
                st.warning(f"Only single letters and numbers are supported: '{word}' skipped.")
                continue

            if os.path.exists(folder_path):
                image_files = glob.glob(os.path.join(folder_path, "*.*"))
                if image_files:
                    st.image(image_files[0], caption=word.capitalize(), use_column_width=True)
                else:
                    st.warning(f"Images not found for '{word}'. Add some in '{folder_path}'.")
            else:
                st.warning(f"Folder for '{word}' not found in: {folder_path}")




