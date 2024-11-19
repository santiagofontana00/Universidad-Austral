import cv2
import csv
import glob
import numpy
import math
import os

from utils.TPFlabel_converters import label_to_int

# Escribo los valores de los momentos de Hu en el archivo
def write_hu_moments(label, writer):
    files = glob.glob('./TPFinal/databaseASL/asl_dataset/' + label + '/*')
    print(f"Procesando {label}: encontrados {len(files)} archivos")
    
    hu_moments = []
    for file in files:
        hu_moments.append(hu_moments_of_file(file))
    for mom in hu_moments:
        flattened = mom.ravel().tolist()  # Convert to Python list
        row = flattened + [label_to_int(label)]  # Combine list and integer
        writer.writerow(row)  # Write to CSV

def generate_hu_moments_file():
    # Asegurar que el directorio existe
    os.makedirs('./TPFinal/TPFgenerated-files', exist_ok=True)
    
    base_path = './TPFinal/databaseASL/asl_dataset'
    
    # Verificar que el directorio existe
    if not os.path.exists(base_path):
        print(f"Error: El directorio {base_path} no existe")
        return
        
    # Obtener las carpetas
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    print(f"Carpetas encontradas: {folders}")
    
    if not folders:
        print("No se encontraron carpetas de letras")
        return

    # Crear o sobreescribir el archivo
    with open('./TPFinal/TPFgenerated-files/TPFhu_moments.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for folder in folders:
            write_hu_moments(folder, writer)


# Encargada de generar los momentos de Hu para las imagenes
def hu_moments_of_file(filename):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 67, 2)

    # Invert the image so the area of the UAV is filled with 1's. This is necessary since
    # cv::findContours describes the boundary of areas consisting of 1's.
    bin = 255 - bin # como sabemos que las figuras son negras invertimos los valores binarios para que esten en 1.

    kernel = numpy.ones((3, 3), numpy.uint8)  # Tamaño del bloque a recorrer
    # buscamos eliminar falsos positivos (puntos blancos en el fondo) para eliminar ruido.
    bin = cv2.morphologyEx(bin, cv2.MORPH_ERODE, kernel)

    contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)  # encuetra los contornos
    shape_contour = max(contours, key=cv2.contourArea)  # Agarra el contorno de area maxima

    # Calculate Moments
    moments = cv2.moments(shape_contour)  # momentos de inercia
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)  # momentos de Hu
    # Log scale hu moments
    for i in range(0, 7):
        huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i])) # Mapeo para agrandar la escala.
    return huMoments
