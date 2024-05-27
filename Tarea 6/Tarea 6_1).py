import tensorflow as tf
from tensorflow.keras.layers import Layer

class capa_color_a_gris(Layer):
    def __init__(self, ):
        super(capa_color_a_gris, self).__init__() 

    def call(self, inputs):
        # Calcula la media ponderada de los canales de color para obtener la escala de grises
        imgris = tf.image.rgb_to_grayscale(inputs) #método de tf
        return imgris
    

from PIL import Image
import tensorflow as tf
import numpy as np

# Ruta de la imagen
ruta = 'Documents\\Github\\Redes-neuronales\\Tarea 6\\cat.10013.jpg'

# Cargar la imagen con PIL
imagen = Image.open(ruta) #leer la imagen y guardarla en la variable

imagen = imagen.resize((512, 512))  # Ajustar el tamaño de la imagen

# Convertir la imagen en un tensor necesitando numpy para darle forato a la func. de conversión
imagen = np.array(imagen) / 255.0  # Normalizar los valores de píxeles al rango [0, 1]
imagen = tf.convert_to_tensor(imagen, dtype=tf.float32)


# creamos un objeto para la capa
img_a_gris=capa_color_a_gris()
# Aplicar la capa
imgris = img_a_gris(imagen) #ya esta la imagen a grises
 
import matplotlib.pyplot as plt

img_gris = imgris.numpy()  # Convierte el tensor a un array NumPy
plt.imshow(img_gris, cmap='gray')  # Muestra la imagen en escala de grises
plt.axis('off') 
plt.show()    