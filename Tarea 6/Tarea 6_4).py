import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Activation
from tensorflow.keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np
     

class EDO(Sequential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse = tf.keras.losses.MeanSquaredError()

    @property
    def metrics(self):
      return [self.loss_tracker]

    def train_step(self, data):
         batch_size = tf.shape(data)[0]
         min = tf.cast(tf.reduce_min(data),tf.float32)
         max = tf.cast(tf.reduce_max(data),tf.float32)
         x = tf.random.uniform((batch_size,1), minval=min, maxval=max)

         with tf.GradientTape() as tape:
             with tf.GradientTape() as tape2:
                 tape2.watch(x)
                 y_pred = self(x, training=True)
             dy = tape2.gradient(y_pred, x) #derivada del modelo con respecto a entradas x
             x_o = tf.zeros((batch_size,1)) #valor de x en condicion inicial x_0=0
             y_o = self(x_o,training=True) #valor del modelo en en x_0
             eq = x*dy + y_pred -tf.pow(x,2.)*tf.cos(x)#Ecuacion diferencial evaluada en el modelo. Queremos que sea muy pequeno
             ic = 0. #valor que queremos para la condicion inicial o el modelo en x_0
             loss = self.mse(0., eq) + self.mse(y_o,ic)

        # Apply grads
         grads = tape.gradient(loss, self.trainable_variables)
         self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #update metrics
         self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
         return {"loss": self.loss_tracker.result()}
     

modelo = EDO()

modelo.add(Dense(10, activation='tanh', input_shape=(1,)))
modelo.add(Dense(1, activation='tanh'))
modelo.add(Dense(1))


modelo.summary()

modelo.compile(optimizer=RMSprop(),metrics=['loss'])

x=tf.linspace(5,5,100)
history = modelo.fit(x,epochs=500,verbose=0)
plt.plot(history.history["loss"])

x_testv = tf.linspace(-5,5,100)
x=np.linspace(-5,5,100)
a=modelo.predict(x_testv)
plt.plot(x_testv,a,label="aprox")
plt.plot(x_testv,((x**2 - 2) * np.sin(x)) / x + 2 * np.cos(x),label="exact")
plt.legend()
plt.show()


class EDO(Sequential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse = tf.keras.losses.MeanSquaredError()

    @property
    def metrics(self):
      return [self.loss_tracker]

    def train_step(self, data):
         batch_size = tf.shape(data)[0]
         min = tf.cast(tf.reduce_min(data),tf.float32)
         max = tf.cast(tf.reduce_max(data),tf.float32)
         x = tf.random.uniform((batch_size,1), minval=min, maxval=max)

         with tf.GradientTape() as tape:
             with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape3:
                    tape3.watch(x)
                    y_pred = self(x, training=True)
                tape2.watch(x)
                dy =tape3.gradient(y_pred, x) #derivada del modelo con respecto a entradas x
             ddy = tape2.gradient(dy, x) #derivada del modelo con respecto a entradas x
             x_o = tf.zeros((batch_size,1)) #valor de x en condicion inicial x_0=0
             y_o = self(x_o,training=True) #valor del modelo en en x_0
             dy_0= self(x_o,training=True) #valor de dy en x_0
             eq = ddy+y_pred#Ecuacion diferencial evaluada en el modelo. Queremos que sea muy pequeno
             ic = 1. #valor que queremos para la condicion inicial o el modelo en x_0
             ic2= -0.5 #valor que queremos para la condicion inicial o el modelo en x_0 para la primera derivada
             loss = self.mse(0., eq) + self.mse(y_o,ic) + self.mse(y_o,ic2)

        # Apply grads
         grads = tape.gradient(loss, self.trainable_variables)
         self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #update metrics
         self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
         return {"loss": self.loss_tracker.result()}
     

modelo = EDO()

modelo.add(Dense(10, activation='tanh', input_shape=(1,)))
modelo.add(Dense(1, activation='tanh'))
modelo.add(Dense(1))


modelo.summary()

modelo.compile(optimizer=Adam(),metrics=['loss'])

x=tf.linspace(5,5,100)
history = modelo.fit(x,epochs=500,verbose=0)
plt.plot(history.history["loss"])

x_testv = tf.linspace(-5,5,100)
x=np.linspace(-5,5,100)
a=modelo.predict(x_testv)
plt.plot(x_testv,a,label="aprox")
plt.plot(x_testv,np.cos(x)-np.sin(x)/2.,label="exact")
plt.legend()
plt.show()

