"""Red densa secuencial con Keras"""

#Importación de librerías
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical


#Configuración de learning rate, número de épocas y batch size
learning_rate = 0.01
epochs = 40
batch_size = 15


#Cargar los datos del conjunto MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#Preprocesamiento de datos del conjunto de imágenes de MNIST
x_trainv = x_train.reshape(60000, 784)  #redimensiona las imágenes de entrenamiento de matrices 28x28 a vectores unidimensionales de longitud 784
x_testv = x_test.reshape(10000, 784)    #redimensiona las imágenes de prueba de matrices 28x28 a vectores unidimensionales de longitud 784
x_trainv = x_trainv.astype('float32')   #cambia el tipo de datos de los valores de los pixeles en las imágenes de prueba y entrenamiento a 'float32' (números de punto flotante)
x_testv = x_testv.astype('float32')

x_trainv /= 255  #Normaliza los valores de los pixeles dividiendo entre 255 escalando los valores de 0 a 1, ya que los originales entán entre 0 y 255. 
x_testv /= 255


#Convertir las etiquetas de los conjuntos de entrenamiento y prueba a formato one-hot encoding
num_classes=10  #define una variable para representar el total de clases (10 porque es una para cada dígito)
y_trainc = to_categorical(y_train, num_classes)  #convierte la etiquetas en formato one-hot encoding usando la función to_categorical
y_testc = to_categorical(y_test, num_classes)



#Creación de la red neuronal
model = Sequential() #crea el objeto de modelo sencuencial en Keras (capas apiladas una encima de la otra)
model.add(Dense(100, activation='sigmoid', input_shape=(784,)))  #agrega una capa densa a la RNA con x neuronas, usa la función de activación sigmoide y tiene una capa de entrada de 784
model.add(Dense(num_classes, activation='sigmoid'))  #segunda capa densa con neuronas = 'num_classes' (generalmente 10) y usa la función de activación sigmoide.


model.summary()  #Imprime un resumen de la arquitectura del modelo


#Compilación del modelo
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(learning_rate=learning_rate),metrics=['accuracy']) 
#configura el entrenamiento de la red especificando la función de pérdida, optimizador y las métricas a usar en el entrenamiento y evaluación de la red, en este caso se usa la 'exactitud'


#Entrenamiento de la red usando los datos de MNIST
#Ajustando los pesos usando el optimizador que se especificó en el model.(compile) y minimizará la función de pérdida
history = model.fit(x_trainv, y_trainc,     #datos de entrada usados para aprender y ajustar los pesos durante el entrenamiento
                    batch_size=batch_size,  #tamaño del batch-size a usar en el entrenamiento
                    epochs=epochs,          #número de épocas
                    verbose=1,              #controla los detalles de lo que se ve durante el entrenamiento, 1 muestra actualizaciones cada época
                    validation_data=(x_testv, y_testc)  #proporciona los datos de validación que evaluan el rendimiento en cada época
                    )



#Evaluación y predicciones del modelo
score = model.evaluate(x_testv, y_testc, verbose=0)  #evalúa el modelo usando los datos de prueba, la variable score guarda los valores de pérdida y la métrica (accuracy)

print('Pérdida en el conjunto de prueba:', score[0]) #Imprime la función de pérdida
print('Precisión en el conjunto de prueba:', score[1]) #Imprime la precisión 