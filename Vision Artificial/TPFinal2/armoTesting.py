import os
import random
import shutil

def move_images_to_testing(source_dir, dest_dir, test_percentage=0.3):
    # Crear el directorio de testing si no existe
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Obtener todas las carpetas de letras
    letter_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for letter_dir in letter_dirs:
        # Obtener lista de imágenes en la carpeta actual
        source_path = os.path.join(source_dir, letter_dir)
        images = [f for f in os.listdir(source_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Calcular cantidad de imágenes para testing
        num_test = int(len(images) * test_percentage)
        
        # Seleccionar aleatoriamente las imágenes para testing
        test_images = random.sample(images, num_test)
        
        # Mover las imágenes seleccionadas
        for image in test_images:
            src = os.path.join(source_path, image)
            new_image_name = f"{letter_dir}_{image}"
            dst = os.path.join(dest_dir, new_image_name)
            shutil.move(src, dst)
            print(f'Movida imagen {image} a {dest_dir}')

if __name__ == "__main__":
    # Especifica aquí las rutas de origen y destino
    source_directory = './TPFinal2/databaseASL/asl_dataset'  # Carpeta que contiene las imágenes originales
    testing_directory = './TPFinal2/databaseASL/TPFtesting'  # Carpeta donde quieres mover las imágenes
    
    move_images_to_testing(
        source_dir=source_directory,
        dest_dir=testing_directory
    )