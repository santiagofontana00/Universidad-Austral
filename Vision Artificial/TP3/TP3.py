import numpy as np
import cv2 as cv

def draw_labels(labels, img):
    # Create a blank white image
    colored_labels = np.ones_like(img) * 255

    # Loop through unique labels (excluding background)
    for label in np.unique(labels):
        if label == 1:
            continue  # Skip background
        # Create a mask for the current label
        mask = (labels == label)
        # Determine color based on area
        area = np.sum(mask)
        area_threshold = cv.getTrackbarPos('Area Threshold', 'Controls')
        color = (255, 0, 0) if area <= area_threshold else (0, 0, 255)  # Red for area <= 50, Blue otherwise
        colored_labels[mask] = color

    cv.imshow('Colored Labels', colored_labels)

def on_trackbar(val):
    pass

# Create named windows
cv.namedWindow('Image')
cv.namedWindow('Controls')

# Create trackbars in the 'Controls' window
cv.createTrackbar('Nucleo', 'Controls', 0, 255, on_trackbar)
cv.createTrackbar('Background', 'Controls', 0, 40, on_trackbar)
cv.createTrackbar('Area Threshold', 'Controls', 5000, 50000, on_trackbar)  # Rango de 0 a 100

while True:
    img = cv.imread("./TP3/levadura.png")
    # Reduce image size
    img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.    INTER_AREA)

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # Get current positions of trackbars
    nucleo = cv.getTrackbarPos('Nucleo', 'Controls')
    background = cv.getTrackbarPos('Background', 'Controls')
    
    # Generate binary image for nucleos
    _, nucleos = cv.threshold(gray, nucleo, 255, cv.THRESH_BINARY)

    # Denoise nucleos
    kernel = np.ones((3,3), np.uint8)
    nucleos_denoised = cv.morphologyEx(nucleos, cv.MORPH_OPEN, kernel, iterations=3)
    nucleos_denoised = cv.morphologyEx(nucleos_denoised, cv.MORPH_CLOSE, kernel, iterations=3)
    
    # Generate binary image for background
    _, background_mask = cv.threshold(gray, background, 255, cv.THRESH_BINARY)

    # Denoise background
    background_denoised = cv.morphologyEx(background_mask, cv.MORPH_OPEN, kernel, iterations=3)
    background_denoised = cv.morphologyEx(background_denoised, cv.MORPH_CLOSE, kernel, iterations=3)

    #Crear el Unknown

    unknown = cv.subtract(background_denoised,nucleos_denoised)


    # Marker labelling
    # Each connected component (foreground object) is assigned a unique label. 
    # The function returns the number of labels and the labeled image
    count, markers = cv.connectedComponents(nucleos_denoised)

    # Add one to all labels so that sure background is not 0, but 1
    # This is done to ensure that the background is labeled as 1 instead of 0. 
    # In watershed segmentation, the background should not be labeled as 0 because 0 is reserved for the unknown region.
    markers = markers+1

    # Now, mark the region of unknown with zero
    # By setting these regions to 0 in the markers image, 
    # we indicate that these regions are to be treated as unknown during the watershed algorithm.
    markers[unknown==255] = 0

    
    # The watershed algorithm treats the markers as seeds and floods the regions to segment the image. 
    # The boundaries of the segmented regions are marked with -1 in the markers image.
    markers = cv.watershed(img,markers)

    # This line marks the boundaries of the segmented regions in the original image img with the color blue 
    img[markers == -1] = [255,0,0]

    # Count the number of unique labels (cells)
    unique_labels = np.unique(markers)
    num_cells = len(unique_labels) - 2  # Subtract 2 to exclude background (1) and boundary (-1) labels

    # Display the number of cells
    cv.putText(img, f'Num Cells: {num_cells}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Dibuja cada celula con un color distinto y muestra el número total de células
    draw_labels(markers, img)

    # Display the original image, denoised nucleos, and denoised background images
    cv.imshow('Image', gray)
    cv.imshow('Nucleos Denoised', nucleos_denoised)
    cv.imshow('Background Denoised', background_denoised)
    cv.imshow('Unknown', unknown)
    cv.imshow('Celulas', img)

    # Create a blank image for controls
    controls = np.zeros((100, 300, 3), dtype=np.uint8)
    cv.putText(controls, f'Nucleo: {nucleo}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv.putText(controls, f'Background: {background}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv.imshow('Controls', controls)
    
    # Wait for a key
    key = cv.waitKey(1) & 0xFF
    
    # If ESC is pressed (ASCII code 27), break the loop
    if key == 27:
        break

cv.destroyAllWindows()

#NO me esta marcando bien las celulas. MI matriz markers me muestra los nucleos. no esta haciendo bien el waterspread sobre la zona unkown.
#hay algo que hace que la linea 73 no ande

# no se que hace la linea 47 de coins.py

#usar 68 y 19 en los trackbats