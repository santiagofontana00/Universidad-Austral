from utils.TPFtesting_model import load_and_test_camera
import cv2

# Load the saved model before testing
model_path = './TPFinal/TPFgenerated-files/trained_model.xml'
try:
    loaded_model = cv2.ml.DTrees_load(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

load_and_test_camera(loaded_model)
