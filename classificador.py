import os 
import tensorflow as tf
import keras
import kagglehub

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout #type:ignore
from tensorflow.keras.images import ImageDataGenerator #type: ignore 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau #type:ignore

arquivoDataset = kagglehub.dataset_download('msambare/fer2013')

dadosTreinamento = os.path.join(arquivoDataset,'train')
dadosTeste = os.path.join(arquivoDataset,'test')

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

classificadorEmocao.summarize()

classificadorEmocao.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


geradorTreinamento = ImageDataGenerator(rescale = 1./255, rotation_range=15,
                                        width_shift_range=0.1, height_shift_range=0.1,
                                        shear_range=0.1, zoom_range=0.2,
                                        horizontal_flip=True, fill_mode='nearest',
                                        validation_split=0.2  )

geradorTeste = ImageDataGenerator(rescale = 1./255)

baseTreinamento = geradorTreinamento.flow_from_directory(dadosTreinamento, target_size=(48,48), color_mode='grayscale',
                                                         batch_size=32, class_mode='categorical',subset = 'training', shuffle=True)

baseTeste = geradorTeste.flow_from_directory(dadosTeste, target_size=(48,48), color_mode = 'grayscale',
                                             batch_size=32, class_mode='categorical',subset = 'training', shuffle=False)

callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.1, patience=5),
    ModelCheckpoint('redeNeuralTccLS.h5', save_best_only=True)
]

classificadorEmocao.fit(baseTreinamento, batch_size = 32, epochs = 100, validation_data = baseTeste, callbacks = callbacks)

lossTest, accTest = classificadorEmocao.evaluate(baseTeste)
print(f"\nAcurácia final no teste: {accTest:.4f}")




