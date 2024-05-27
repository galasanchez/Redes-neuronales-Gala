import zipfile
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from keras.api._v2.keras import regularizers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Activation, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import mlflow
import torch

# Extract zip file
with zipfile.ZipFile(r'C:\Users\LAP HP\Documents\ESCUELA\Redes_Neuronales\DogsCats.zip') as zip_ref:
    zip_ref.extractall('/content')

# Initialize Comet ML Experiment
experiment = Experiment(
    api_key="kWkfZpAz7NFyurSYJuLDm8QXA",
    project_name="perrosygatos",
    workspace="galasanchez"
)

experiment.set_name('experimento')

ih, iw = 150, 150  # tamaño de la imagen
input_shape = (ih, iw, 3)  # forma de la imagen: alto, ancho, y número de canales

train_dir = 'train'  # directorio de entrenamiento
test_dir = 'test'  # directorio de prueba

num_class = 2  # cuántas clases
epochs = 60  # cuántas veces entrenar. En cada epoch hace una mejora en los parámetros
batch_size = 100  # batch para hacer cada entrenamiento. Lee 'batch_size' imágenes antes de actualizar los parámetros. Las carga a memoria
num_train = 2000  # número de imágenes en train
num_test = 1200  # número de imágenes en test

epoch_steps = num_train // batch_size
test_steps = num_test // batch_size

# Report multiple hyperparameters using a dictionary:
hyper_params = {
    "epochs": epochs,
    "steps": epoch_steps,
    "batch_size": batch_size,
}
experiment.log_parameters(hyper_params)

# Initialize and train your model
gentrain = ImageDataGenerator(rescale=1. / 255.,  # indica que reescale cada canal con valor entre 0 y 1.
                              rotation_range=40,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True,
                              fill_mode='nearest')

train = gentrain.flow_from_directory(train_dir,
                                     batch_size=batch_size,
                                     target_size=(iw, ih),
                                     class_mode='binary')

gentest = ImageDataGenerator(rescale=1. / 255)

test = gentest.flow_from_directory(test_dir,
                                   batch_size=batch_size,
                                   target_size=(iw, ih),
                                   class_mode='binary')

# para cargar pesos de la red desde donde se quedó la última vez
# filename = "cvsd.h5"
# model.load_weights(filename)  # comentar si se comienza desde cero.
###
parameters = {
    "batch_size": batch_size,
    "epochs": epochs,
    "optimizer": 'adam',
    "loss": 'sparse_categorical_crossentropy',
}

experiment.log_parameters(parameters)

model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(ih, iw, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.summary()

filepath = "m_pg_1.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model.compile(loss=parameters['loss'],
              optimizer=parameters['optimizer'],
              metrics=['accuracy'])

model.fit(train,
          steps_per_epoch=epoch_steps,
          epochs=epochs,
          validation_data=test,
          validation_steps=test_steps,
          verbose=1,
          callbacks=[checkpoint])

experiment.log_model("Perros_y_Gatos", "m_pg_1.hdf5")
experiment.end()
model.save('cvsd.h5')
