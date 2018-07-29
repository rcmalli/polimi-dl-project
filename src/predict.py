from model import depth_model
import tensorflow as tf
import numpy as np
from utils import get_args
from config import process_config
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
from model import  load_depth_model_from_weights


def predict():

    args = get_args()
    config = process_config(args.config)
    model = load_depth_model_from_weights(config)


    for i in range(6):

        img = image.load_img('../images/'+str(i+1) + '.jpg', target_size=(config.input_size[0], config.input_size[1]))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        prediction = model.predict(x)
        print("prediction shape",prediction.shape)
        prediction = np.reshape(prediction, [prediction.shape[1], prediction.shape[2]])
        plt.imsave('../images/depth_'+str(i+1) + '.jpg', prediction)



if __name__ == '__main__':
    predict()