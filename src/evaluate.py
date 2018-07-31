from data import load_pair_paths, tf_data_generator, split_dataset, calculate_num_iter
from model import depth_model, load_depth_model_from_weights
from loss import select_loss, mean_absolute_error,mean_squared_error
from metric import root_mean_squared_error, abs_relative
from dirs import create_dirs
import os
import tensorflow as tf
from train import select_optimizer
from utils import get_args
from config import process_config
from tensorflow.keras.optimizers import SGD, Adam



def extract(dataset, config):

    model = depth_model(config)
    config.output_size = list(model.output_shape[1:])
    model.compile(optimizer=select_optimizer(config), loss=select_loss(config),
                  metrics=[mean_absolute_error, mean_squared_error,root_mean_squared_error,abs_relative])
    model.load_weights(config.model_dir + config.prediction_model_name)

    # split dataset train and test
    train_pairs, test_pairs = split_dataset(config, dataset)
    test_num_steps = calculate_num_iter(config, test_pairs)
    test_gen = tf_data_generator(config, test_pairs, is_training=False)
    result = model.evaluate(test_gen, steps=test_num_steps, verbose=1)
    tf.keras.backend.clear_session()
    return model.metrics_names, result

def evaluate():

    args = get_args()
    config = process_config(args.config)
    # load dataset file
    dataset = load_pair_paths(config)

    metric_names = []
    results = []
    model_names = []

    config.unpool_type = "simple"
    config.exp_name = "nyu-resnet-berhu-aug-30-simple-upproject"
    config.prediction_model_name = "model-150-0.19.km"
    config.model_dir = os.path.join("../experiments", config.exp_name, "model/")
    config.tensorboard_dir = os.path.join("../experiments", config.exp_name, "log/")

    metric_names, result = extract(dataset, config)
    results.append(result)
    model_names.append(config.unpool_type + "_"+ config.model_type+ "_" + config.loss_type)

    config.unpool_type = "deconv"
    config.exp_name = "nyu-resnet-berhu-aug-30-deconv-upproject"
    config.prediction_model_name = "model-150-0.21.km"
    config.model_dir = os.path.join("../experiments", config.exp_name, "model/")
    config.tensorboard_dir = os.path.join("../experiments", config.exp_name, "log/")

    metric_names, result = extract(dataset, config)
    results.append(result)
    model_names.append(config.unpool_type + "_"+ config.model_type+ "_" + config.loss_type)

    config.unpool_type = "checkerboard"
    config.exp_name = "nyu-resnet-berhu-aug-30-checkerboard-upproject"
    config.prediction_model_name = "model-150-0.20.km"
    config.model_dir = os.path.join("../experiments", config.exp_name, "model/")
    config.tensorboard_dir = os.path.join("../experiments", config.exp_name, "log/")

    metric_names, result = extract(dataset, config)
    results.append(result)
    model_names.append(config.unpool_type + "_"+ config.model_type+ "_" + config.loss_type)

    config.unpool_type = "resize"
    config.exp_name = "nyu-resnet-berhu-aug-30-resize-upproject"
    config.prediction_model_name = "model-150-0.20.km"
    config.model_dir = os.path.join("../experiments", config.exp_name, "model/")
    config.tensorboard_dir = os.path.join("../experiments", config.exp_name, "log/")

    metric_names, result = extract(dataset, config)
    results.append(result)
    model_names.append(config.unpool_type + "_"+ config.model_type+ "_" + config.loss_type)

    print(metric_names)
    print(results)
    print(model_names)


if __name__ == '__main__':
    evaluate()