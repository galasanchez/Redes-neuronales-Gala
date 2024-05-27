import os
import tensorflow as tf
from tensorflow.keras.layers import Layer
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class capa_color_a_gris(Layer):
    def __init__(self):
        super(capa_color_a_gris, self).__init__()

    def call(self, inputs):
        imgris = tf.image.rgb_to_grayscale(inputs)
        return imgris

# Construir la ruta de la imagen
base_path = 'Documents'
filename = 'cat.10013.jpg'
ruta = os.path.join(base_path, filename)

# Verificar si la imagen existe
if not os.path.exists(ruta):
    print(f"La imagen en la ruta {ruta} no existe.")
else:
    # Cargar la imagen con PIL
    imagen = Image.open(ruta)
    imagen = imagen.resize((512, 512))  # Ajustar el tamaño de la imagen

    # Convertir la imagen en un tensor necesitando numpy para darle formato a la función de conversión
    imagen = np.array(imagen) / 255.0  # Normalizar los valores de píxeles al rango [0, 1]
    imagen = tf.convert_to_tensor(imagen, dtype=tf.float32)
    
    # Añadir una dimensión para el batch
    imagen = tf.expand_dims(imagen, axis=0)

    # Crear un objeto para la capa
    img_a_gris = capa_color_a_gris()

    # Aplicar la capa
    imgris = img_a_gris(imagen)

    # Convertir la imagen procesada a un array NumPy y eliminar la dimensión del batch
    img_gris = tf.squeeze(imgris).numpy()

    # Mostrar la imagen en escala de grises
    plt.imshow(img_gris, cmap='gray')
    plt.axis('off')
    plt.show()
