import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from matplotlib import pyplot as plt
import numpy as np
import math

loss_tracker = keras.metrics.Mean(name="loss") #actualiza la media de métricas relevantes en el rendimiento del entrenamiento
#establecer metodos para la red a partir de su estructura (model.sequential)
class SolucionarEc(Sequential):
    @property
    def metrics(self):
        return [loss_tracker] #cambia el loss_tracker
#entrenar la red
    def train_step(self, data):
        batch_size =100 #Calibra la resolucion, define los puntos a entrenar en la gráfica
        x = tf.random.uniform((batch_size,1), minval=-1, maxval=1) #definir los puntos y el dominio
        f = 1.+2.*x+4.*x**3 #función a aproximar (etiquetas para el conjunto de datos de entrada x)

        #calculo de las parciales (y por ende la fución de costo)
        #modificamos el algoritmo de calculo dde los gradientes para hacerlo funcionar en red personalizada
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True) #calcular las predicciones (y hat)

            # Usando 'sum'
            mse = tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.SUM)
            loss= mse(f, y_pred)

        grads = tape.gradient(loss, self.trainable_weights) #calcular las parcales (los gradientes de w y b)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))#aplicar la optimización (?)
        #actualiza metricas
        loss_tracker.update_state(loss)

        return {"loss": loss_tracker.result()}
    
#crear instancia para el entrenamiento y la estructura de la red
modelo = SolucionarEc()
modelo.add(Dense(600,activation='tanh', input_shape=(1,)))
modelo.add(Dense(400,activation='tanh'))
modelo.add(Dense(300,activation='tanh'))
modelo.add(Dense(200,activation='tanh'))
modelo.add(Dense(100,activation='relu'))
modelo.add(Dense(1))

modelo.summary()

modelo.compile(optimizer=Adam(learning_rate=0.0001), metrics=['loss'])
     

x=tf.linspace(-1,1,100) #cargar datos de entrada para la red una vez entrenada (el rango de la función)
     

historial = modelo.fit(x,epochs=10000,verbose=0) #entrenar la red
     

plt.plot(historial.history["loss"])
     
#probar la red
prueba=modelo.predict(x) #predicción para cada dato de entrada
     

plt.plot(x,prueba,label="Predicción") #graficar la predicción
plt.plot(x, 1.+2.*x+4.*x**3, label="Real") #graficar la función real

plt.legend()
plt.grid()
plt.show()