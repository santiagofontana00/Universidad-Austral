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
        
        # Navegación actualizada:
        # Flecha derecha o 'd': siguiente imagen
        # Flecha izquierda o 'a': imagen anterior
        # 'f': saltar 21 imágenes adelante
        # 'q': salir
        if key == ord('q'):
            break
        elif key in [ord('d'), 83]:  # Derecha
            current_index = (current_index + 1) % total_images
        elif key in [ord('a'), 81]:  # Izquierda
            current_index = (current_index - 1) % total_images
        elif key == ord('f'):  # Saltar 21 imágenes
            current_index = (current_index + 21) % total_images

def calculate_accuracy_per_letter():
    # Diccionario para almacenar aciertos por letra
    correct_predictions = {chr(i): 0 for i in range(97, 123)}  # a-z
    
    for file in files:
        # Obtener la letra real del nombre del archivo (tomando solo el nombre del archivo, no la ruta completa)
        filename = file.split('\\')[-1]  # o file.split('\\')[-1] en Windows
        true_letter = filename[0].lower()
        
        # Calcular predicción
        hu_moments = hu_moments_of_file(file)
        sample = np.array([hu_moments], dtype=np.float32)
        prediction = loaded_model.predict(sample)[1]
        predicted_letter = int_to_label(prediction).lower()
        
        # Contar aciertos
        if predicted_letter == true_letter:
            correct_predictions[true_letter] += 1
    
    # Imprimir resultados
    print("\nPrecisión por letra:")
    print("-" * 30)
    high_accuracy_letters = []
    for letter in correct_predictions:
        accuracy = (correct_predictions[letter] / 21) * 100
        print(f"Letra {letter.upper()}: {accuracy:.2f}%")
        if accuracy > 80:
            high_accuracy_letters.append((letter, accuracy))
    
    # Imprimir letras con alta precisión
    print("\nLetras con precisión mayor al 80%:")
    print("-" * 30)
    for letter, accuracy in high_accuracy_letters:
        print(f"Letra {letter.upper()}: {accuracy:.2f}%")

# Cargar el modelo entrenado
model_path = './TPFinal2/TPFgenerated-files/trained_model.xml'
try:
    loaded_model = cv2.ml.DTrees_load(model_path)
    print("Modelo cargado exitosamente")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# Obtener todas las imágenes de la letra 'a'
files = sorted(glob.glob('./TPFinal2/databaseASL/TPFtesting/*'))

if not files:
    print("No se encontraron imágenes en la carpeta especificada")
    exit()

calculate_accuracy_per_letter()
show_images(files)
cv2.destroyAllWindows()

