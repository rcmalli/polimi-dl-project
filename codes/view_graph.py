import tensorflow as tf
import tensorflow_hub as hub


with tf.Session() as sess:
    #module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/1")
    latest_checkpoint = tf.train.latest_checkpoint("./Data/ResnetNYU/")
    saver = tf.train.Saver()
    saver.restore(sess, latest_checkpoint)
    graph = tf.get_default_graph()
LOGDIR='../experiments/GRAPH-TEST'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)