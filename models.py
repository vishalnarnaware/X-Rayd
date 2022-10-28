import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import tensorflow.keras as K
import matplotlib.pyplot as plt
#from skimage.transform import resize
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import Model
from tensorflow.keras import layers
from explain import VizGradCAM
#from classification_models.tfkeras import Classifiers
import matplotlib
matplotlib.use('Agg')


class TB:
    def __init__(self, modelName=None):
        if modelName==None:
            self.modelName = 'full_tb_model.h5'
        else:
            self.modelName = modelName

        self.model = tf.keras.models.load_model(os.path.join('models', self.modelName))

    def predict(self, img_path):
        self.test_img = img_to_array(load_img(img_path , target_size=(224,224)))
        img = np.expand_dims(self.test_img, axis=0)
        pred = self.model.predict(img)
        return pred[0][0]

    def explain(self, ):
        VizGradCAM(self.model, self.test_img, 'explain_tb.png')

class Cancer:
    def __init__(self, modelName=None):
        if modelName==None:
            self.modelName = 'full_cancer_model_m.h5'
        else:
            self.modelName = modelName

        self.model = tf.keras.models.load_model(os.path.join('models', self.modelName))

    def predict(self, img_path):
        self.test_img = img_to_array(load_img(img_path , target_size=(224,224)))
        img = np.expand_dims(self.test_img, axis=0)
        pred = self.model.predict(img)
        return pred[0]

    def explain(self, ):
        VizGradCAM(self.model, self.test_img, 'explain_can.png')


class Covid:
    def __init__(self, modelName=None):
        if modelName==None:
            self.modelName = 'full_covid_model.h5'
        else:
            self.modelName = modelName

        self.model = tf.keras.models.load_model(os.path.join('models', self.modelName))

    def predict(self, img_path):
        self.test_img = img_to_array(load_img(img_path , target_size=(224,224)))
        img = np.expand_dims(self.test_img, axis=0)
        pred = self.model.predict(img)
        return pred[0]

    def explain(self, ):
        VizGradCAM(self.model, self.test_img, 'explain_cov.png')


class Multiple:
    def __init__(self, modelName=None):
        if modelName==None:
            self.modelName = 'full_multi_model.h5'
        else:
            self.modelName = modelName

        def build_model():
            seresnet152, _ = Classifiers.get('seresnet152')

            inp = layers.Input(shape=(600, 600, 3))
            pretrained_base = seresnet152(input_shape=(600, 600, 3),
                                          include_top=False,
                                          input_tensor=inp,
                                          weights='imagenet')
            conv2d_254 = pretrained_base.get_layer('conv2d_254')
            x = layers.Dropout(0.16)(conv2d_254.output)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(14, 'sigmoid')(x)
            return Model(inp, x)
        self.model = build_model()
        self.model.load_weights(os.path.join('models', self.modelName), by_name=True)
        #self.model = tf.keras.models.load_model(os.path.join('models', self.modelName))

    def predict(self, img_path):
        self.test_img = img_to_array(load_img(img_path , target_size=(600,600)))
        img = np.expand_dims(self.test_img, axis=0)
        pred = self.model.predict(img)
        return pred[0]

    def explain(self, ):
        VizGradCAM(self.model, self.test_img, 'explain_mult.png')

if __name__ == '__main__':
    tb = Cancer()
    print(tb.predict(os.path.join(os.getcwd(), 'static', 'inference', '000016.png')))
    tb.explain()




