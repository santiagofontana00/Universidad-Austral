import cv2
import glob
import numpy as np
from utils.TPFhu_moments_generation import hu_moments_of_file
from utils.TPFlabel_converters import int_to_label

def show_images(files):
    current_index = 0
    total_images = len(files)

    while True:
        # Leer la imagen actual
        image = cv2.imread(files[current_index])
        
        # Obtener el nombre del archivo para mostrarlo
        filename = files[current_index].split('/')[-1]
        
        # Preparar imagen binaria (similar al proceso en hu_moments_generation)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
        
        # Calcular los momentos de Hu y predecir
        hu_moments = hu_moments_of_file(files[current_index])
        sample = np.array([hu_moments], dtype=np.float32)
        prediction = loaded_model.predict(sample)[1]
        predicted_label = int_to_label(prediction)
        
        # Agregar textos a la imagen original
        image = cv2.putText(
            image, 
            f"Predicción: {predicted_label}", 
            (20, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2, 
            cv2.LINE_AA
        )
        
        image = cv2.putText(
            image,
            f"Imagen {current_index + 1}/{total_images}: {filename}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        
        # Mostrar ambas imágenes
        cv2.imshow("ASL Dataset Test", image)
        cv2.imshow("Binary Image", binary)
        
        # Esperar por una tecla
        key = cv2.waitKey(0)
        
        # Navegación:
        # Flecha derecha o 'd': siguiente imagen
        # Flecha izquierda o 'a': imagen anterior
        # 'q': salir
        if key == ord('q'):
            break
        elif key in [ord('d'), 83]:  # Derecha
            current_index = (current_index + 1) % total_images
        elif key in [ord('a'), 81]:  # Izquierda
            current_index = (current_index - 1) % total_images

# Cargar el modelo entrenado
model_path = './TPFinal/TPFgenerated-files/trained_model.xml'
try:
    loaded_model = cv2.ml.DTrees_load(model_path)
    print("Modelo cargado exitosamente")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# Obtener todas las imágenes de la letra 'a'
files = sorted(glob.glob('./TPFinal/databaseASL/asl_dataset/o/*'))

if not files:
    print("No se encontraron imágenes en la carpeta especificada")
    exit()

show_images(files)
cv2.destroyAllWindows()
