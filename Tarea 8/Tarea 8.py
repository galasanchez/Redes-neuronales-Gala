from google.colab import drive

# Ruta para trabajar desde drive
drive.mount('/content/drive')

     
Mounted at /content/drive

!unzip -qq /content/drive/MyDrive/Datasets/img_align_celeba.zip
     

from PIL import Image
from IPython.display import display

# Ruta a la imagen en Google Drive
ruta_imagen = '/content/img_align_celeba/202599.jpg'


# Cargar la imagen con PIL
imagen = Image.open(ruta_imagen)

# Mostrar la imagen
display(imagen)


# Crear df con las propiedades necesarias
import pandas as pd

archivo_txt = '/content/drive/MyDrive/Datasets/list_attr_celeba.txt'
df1=pd.read_csv(archivo_txt, delim_whitespace=True)

print(df1.head(10))
     
import numpy as np

columnas=np.array(df1.columns)
df1[columnas[1:]]=df1[columnas[1:]].replace({-1:0,1:1})
df1[columnas[0]]='/content/img_align_celeba/'+df1[columnas[0]]

print(df1.head(3))

print(df1.iloc[:2000,0])
     
from tensorflow.keras.preprocessing.image import ImageDataGenerator
ih,iw=180,180

dataconf= ImageDataGenerator(rescale=1./255, validation_split=0.4)

dataset_tr= dataconf.flow_from_dataframe(
    dataframe=df1.head(40000),
    x_col='filename',
    y_col=df1.columns[1:],
    target_size=(ih,iw),
    batch_size=100,
    class_mode='raw',
    subset='training')

dataset_vl= dataconf.flow_from_dataframe(
    dataframe=df1.head(40000),
    x_col='filename',
    y_col=df1.columns[1:],
    target_size=(ih,iw),
    batch_size=100,
    class_mode='raw',
    subset='validation')


import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.api._v2.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint



modelf=Sequential()

modelf.add(Conv2D(64, (3, 3), input_shape=(ih, iw,3),kernel_initializer="glorot_uniform"))#,padding="valid",kernel_initializer="glorot_uniform",kernel_regularizer=L1(0.01)))
modelf.add(BatchNormalization())
modelf.add(Activation('relu'))
modelf.add(MaxPooling2D((2, 2)))

modelf.add(Conv2D(128, (3, 3),kernel_initializer="glorot_uniform"))#,padding="valid",kernel_initializer="glorot_uniform",kernel_regularizer=L1(0.01)))
modelf.add(BatchNormalization())
modelf.add(Activation('relu'))
modelf.add(MaxPooling2D(pool_size=(2, 2)))

modelf.add(Conv2D(256, (3, 3),kernel_initializer="glorot_uniform"))#,padding="valid",kernel_initializer="glorot_uniform",kernel_regularizer=L1(0.01)))
modelf.add(BatchNormalization())
modelf.add(Activation('relu'))
modelf.add(MaxPooling2D(pool_size=(2, 2)))

modelf.add(Conv2D(512, (3, 3),kernel_initializer="glorot_uniform"))#,padding="valid",kernel_initializer="glorot_uniform",kernel_regularizer=L1(0.01)))
modelf.add(BatchNormalization())
modelf.add(Activation('relu'))
modelf.add(MaxPooling2D(pool_size=(2, 2)))


modelf.add(Flatten())
modelf.add(Dense(256,activation='relu'))
modelf.add(Dense(40,activation='sigmoid'))

modelf.summary()

ruta_check= '/content/drive/MyDrive/Datasets/caras_checkpoints'
checkpoint = ModelCheckpoint(ruta_check, save_best_only=True)

opt=Adam(learning_rate=0.1)
modelf.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=opt)

earlycall=callbacks.EarlyStopping(monitor = "loss", patience = 3, mode = "auto", restore_best_weights=True)


history = modelf.fit(dataset_tr,
                     epochs=10,
                     batch_size=1000,
                     steps_per_epoch=40,
                     validation_data=dataset_vl,
                     callbacks=[checkpoint],
                     verbose=1)

     
import matplotlib.pyplot as plt
#graficar entrenamiento
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.plot(histdil.history['accuracy'])

plt.xlabel('Epoca')
plt.ylabel('binary_crossentropy')
plt.title('Funciónde Costo en el tiempo')
plt.legend(['Entrenamiento','Validación'])
plt.show()
     


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
a=Image.open('/content/img_align_celeba/202599.jpg').resize((180,180))
a=image.img_to_array(a)
a=preprocess_input(a.reshape(1,180,180,3))
a=modelf.predict(a)
a=a.reshape(40,)
respuestas=[round(x) for x in a]
print(respuestas)