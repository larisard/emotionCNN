import os
import tensorflow as tf

from tensorflow.keras.models import load_model #type: ignore

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

classificadorEmocao = load_model('model_final.keras')

