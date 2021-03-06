from data import load_pair_paths, tf_data_generator, split_dataset, calculate_num_iter
from model import depth_model
from loss import select_loss
from dirs import create_dirs

from utils import get_args
from config import process_config
from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def create_callbacks(config):
    #es = EarlyStopping(monitor='val_loss', patience=config.early_stop_patience)
    tb = TensorBoard(log_dir=config.tensorboard_dir, write_images=True)
    rp = ReduceLROnPlateau(monitor='val_loss', factor=config.reduce_lr_factor,verbose=1, min_lr=0.0000001)
    mc = ModelCheckpoint(filepath=config.model_dir + 'model-{epoch:02d}-{val_loss:.2f}.km', save_weights_only=True,
                         period=15, verbose=1)
    return [tb, rp, mc]


def select_optimizer(config):
    if config.optimizer == "ADAM":
        return Adam(lr=config.learning_rate)
    else:
        return SGD(lr=config.learning_rate)


def train():
    # load config file and prepare experiment
    args = get_args()
    config = process_config(args.config)
    create_dirs([config.model_dir, config.tensorboard_dir])

    # load dataset file
    dataset = load_pair_paths(config)

    # split dataset train and test
    train_pairs, test_pairs = split_dataset(config, dataset)

    if config.debug:
        print("WARNING!!! DEBUG MODE ON! 100 training.")
        train_pairs = train_pairs[:100]
        print(train_pairs)
        test_pairs = test_pairs[:100]
        print(test_pairs)

    # Calculate steps for each epoch
    train_num_steps = calculate_num_iter(config, train_pairs)
    test_num_steps = calculate_num_iter(config, test_pairs)


    # Create the model
    model = depth_model(config)

    #set dynamic output shape
    config.output_size = list(model.output_shape[1:])

    # Create train and test data generators
    train_gen = tf_data_generator(config, train_pairs, is_training=True)
    test_gen = tf_data_generator(config,test_pairs, is_training=False)

    # Prepare for training
    model.compile(optimizer=select_optimizer(config), loss=select_loss(config))


    model.fit(
        train_gen,
        steps_per_epoch=train_num_steps,
        epochs=config.num_epochs,
        callbacks=create_callbacks(config),
        validation_data=test_gen,
        validation_steps=test_num_steps,
        verbose=1)



    print("Training Done.")


if __name__ == '__main__':
    train()
