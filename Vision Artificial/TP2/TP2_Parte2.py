from utils.testing_model import load_and_test_camera
import cv2

# Load the saved model before testing
model_path = './TP2/generated-files/trained_model.xml'
try:
    loaded_model = cv2.ml.DTrees_load(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

load_and_test_camera(loaded_model)

#loaded_model = cv2.ml.DTrees_load(model_path)
#load_and_test(loaded_model)
#
#hay que ver que pasa si en el archivo de entrenamiento hay mas de una imagen

#el tercero es el de match shapes(?. Esto quiere decir que hay que ponerlo a andar como el tp1, para un observador ajeno al tema se va  a ver igual. PEro nosotros sabemos que es distinto, por ejemplo, no hay unknowns en este caso, y en el tp1 si.



#el csv no tiene que ser con nombres sino con labels, no tiene que ser estrella, cuadrado,etc sino 0,1,2,3,etc
