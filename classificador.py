import os 
import tensorflow as tf
import keras
import kagglehub
import pickle
import cv2
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout #type:ignore
from tensorflow.keras.images import ImageDataGenerator #type: ignore 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau #type:ignore
from skimage.feature import hog

arquivoDataset = kagglehub.dataset_download('msambare/fer2013')

dadosTreinamento = os.path.join(arquivoDataset,'train')
dadosTeste = os.path.join(arquivoDataset,'test')


def extrair_hog_img(imagens):
    imagens_hog = []
    for img in imagens:
        imagem = img.reshape(48, 48)
        _, hog_image = hog(imagem,
                           orientations=9,
                           pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2),
                           visualize=True,
                           feature_vector=False)
        
        hog_image = cv2.resize(hog_image, (48, 48))
        imagens_hog.append(hog_image)
    return np.expand_dims(np.array(imagens_hog), axis=-1)  

geradorTreinamento = ImageDataGenerator(rescale = 1./255, rotation_range=15,
                                        width_shift_range=0.1, height_shift_range=0.1,
                                        shear_range=0.1, zoom_range=0.2,
                                        horizontal_flip=True, fill_mode='nearest',
                                        validation_split=0.2  )

geradorTeste = ImageDataGenerator(rescale = 1./255)

baseTreinamento = geradorTreinamento.flow_from_directory(dadosTreinamento, target_size=(48,48), color_mode='grayscale',
                                                         batch_size=32, class_mode='categorical',subset = 'training', shuffle=True)

X_treino, y_treino = baseTreinamento.next()

baseTeste = geradorTeste.flow_from_directory(dadosTeste, target_size=(48,48), color_mode = 'grayscale',
                                             batch_size=32, class_mode='categorical',subset = 'training', shuffle=False)

X_teste, y_teste = baseTeste.next()


X_treino_hog = extrair_hog_img(X_treino)
X_teste_hog = extrair_hog_img(X_teste)

classificadorEmocao =  Sequential()

classificadorEmocao.add(InputLayer(shape = (48,48,1)))

classificadorEmocao.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
classificadorEmocao.add(MaxPooling2D(pool_size = (2,2)))

classificadorEmocao.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
classificadorEmocao.add(MaxPooling2D(pool_size = (2,2)))

classificadorEmocao.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
classificadorEmocao.add(MaxPooling2D(pool_size = (2,2)))

classificadorEmocao.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
classificadorEmocao.add(MaxPooling2D(pool_size = (2,2)))

classificadorEmocao.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
classificadorEmocao.add(MaxPooling2D(pool_size = (2,2)))

classificadorEmocao.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
classificadorEmocao.add(MaxPooling2D(pool_size = (2,2)))

classificadorEmocao.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
classificadorEmocao.add(MaxPooling2D(pool_size = (2,2)))

classificadorEmocao.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
classificadorEmocao.add(MaxPooling2D(pool_size = (2,2)))

classificadorEmocao.add(Flatten())

classificadorEmocao.add(Dense(units = 128, activation = 'relu', ))
classificadorEmocao.add(Dropout(0.2))

classificadorEmocao.add(Dense(units = 128, activation = 'relu', ))
classificadorEmocao.add(Dropout(0.25))

classificadorEmocao.add(Dense(units = 128, activation = 'relu', ))
classificadorEmocao.add(Dropout(0.3))

classificadorEmocao.add(Dense(units = 7, activation ='softmax'))

classificadorEmocao.summary()

classificadorEmocao.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.1, patience=5),
    ModelCheckpoint('redeNeuralTccLS.h5', save_best_only=True)
]

classificadorEmocao.fit(X_treino_hog, y_treino, batch_size = 32, epochs = 100, validation_data = (X_teste_hog, y_teste), callbacks = callbacks)

lossTest, accTest = classificadorEmocao.evaluate(baseTeste)
print(f"\nAcurácia final no teste: {accTest:.4f}")


classificadorEmocao.save('modelo_final.keras')


