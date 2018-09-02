from model import depth_model
import tensorflow as tf
import numpy as np
from utils import get_args
from config import process_config, get_config_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
from model import  load_depth_model_from_weights
from dirs import  create_dirs
import  os
import pickle
from PIL import Image

gt_path = "../images/dataset_gt"
d_img_path = "../images/dataset_img"
e_img_path = "../images/external_img"

def create_outputs():

    args = get_args()
    config, _ = get_config_from_json(args.config)
    dir = "../outputs/input"
    create_dirs([dir])

    for i in range(6):
        img = Image.open(e_img_path + "/" + str(i + 1) + '.jpg')
        img = img.resize((config.input_size[1], config.input_size[0]))
        x = np.array(img)
        plt.imsave(dir + '/ext_' + str(i + 1) + '.jpg', x )

    for i in range(6):
        img = Image.open(d_img_path + "/" + str(i + 1) + '.jpg')
        img = img.resize((config.input_size[1], config.input_size[0]))
        x = np.array(img)
        plt.imsave(dir + '/dat_' + str(i + 1) + '.jpg', x)



    config.unpool_type = "simple"
    config.exp_name = "nyu-resnet-berhu-aug-30-simple-upproject"
    config.prediction_model_name = "model-150-0.19.km"
    config.model_dir = os.path.join("../experiments", config.exp_name, "model/")
    config.tensorboard_dir = os.path.join("../experiments", config.exp_name, "log/")

    extract(config)

    tf.keras.backend.clear_session()

    config.unpool_type = "deconv"
    config.exp_name = "nyu-resnet-berhu-aug-30-deconv-upproject"
    config.prediction_model_name = "model-150-0.21.km"
    config.model_dir = os.path.join("../experiments", config.exp_name, "model/")
    config.tensorboard_dir = os.path.join("../experiments", config.exp_name, "log/")

    extract(config)

    tf.keras.backend.clear_session()

    config.unpool_type = "checkerboard"
    config.exp_name = "nyu-resnet-berhu-aug-30-checkerboard-upproject"
    config.prediction_model_name = "model-150-0.20.km"
    config.model_dir = os.path.join("../experiments", config.exp_name, "model/")
    config.tensorboard_dir = os.path.join("../experiments", config.exp_name, "log/")

    extract(config)

    tf.keras.backend.clear_session()


    config.unpool_type = "resize"
    config.exp_name = "nyu-resnet-berhu-aug-30-resize-upproject"
    config.prediction_model_name = "model-150-0.20.km"
    config.model_dir = os.path.join("../experiments", config.exp_name, "model/")
    config.tensorboard_dir = os.path.join("../experiments", config.exp_name, "log/")

    extract(config)




def extract(config):

    model = load_depth_model_from_weights(config)
    dir = "../outputs/" + config.exp_name
    create_dirs([dir])

    for i in range(6):

        img = image.load_img(e_img_path+"/"+str(i+1) + '.jpg', target_size=(config.input_size[0], config.input_size[1]))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        prediction = model.predict(x)
        print("prediction shape", prediction.shape)
        prediction = np.reshape(prediction, [prediction.shape[1], prediction.shape[2]])
        plt.imsave(dir + '/ext_pre_depth_'+str(i+1) + '.jpg', prediction)

    for i in range(6):

        img = image.load_img(d_img_path+"/"+str(i+1) + '.jpg', target_size=(config.input_size[0], config.input_size[1]))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        prediction = model.predict(x)
        print("prediction shape", prediction.shape)
        prediction = np.reshape(prediction, [prediction.shape[1], prediction.shape[2]])
        plt.imsave(dir + '/dat_pre_depth_'+str(i+1) + '.jpg', prediction)

    for i in range(6):

        with open(gt_path+"/"+str(i+1) + '.pkl', 'rb') as fp:
            depth = pickle.load(fp)/10.0
            depth = Image.fromarray(depth)
            depth = np.array(depth.resize((160, 112)))
            plt.imsave(dir + '/dat_gt_depth_' + str(i + 1) + '.jpg', depth)

    del model



if __name__ == '__main__':
    create_outputs()