from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf

class DepthTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(DepthTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        # This function implements the operation in one epoch

        self.sess.run(self.data.iterator.initializer, feed_dict={self.data.image: self.data.train_images,
                                                                 self.data.depth: self.data.train_depths})

        train_loop = tqdm(range(self.config.train_num_iter_per_epoch))
        lose_list = []
        acc_list = []

        # Loop for the batch
        for _ in train_loop:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                loss, acc = self.train_step()
                lose_list.append(loss)
                acc_list.append(acc)

        train_loss = np.mean(lose_list)
        train_acc = np.mean(acc_list)

        #test loop

        self.sess.run(self.data.iterator.initializer, feed_dict={self.data.image: self.data.test_images,
                                                                 self.data.depth: self.data.test_depths})

        test_loop = tqdm(range(self.config.test_num_iter_per_epoch))
        lose_list = []
        acc_list = []
        image_out = [] # it will be overwritten
        # Loop for the batch
        for _ in test_loop:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                loss, acc = self.test_step()
                lose_list.append(loss)
                acc_list.append(acc)

        test_loss = np.mean(lose_list)
        test_acc = np.mean(acc_list)


        # Epoch summary
        cur_it = self.model.global_step_tensor.eval(self.sess)



        # For observing additional parameters, it is enough to add it to summaries_dict with a proper tag
        summaries_dict = {
            'train_loss': train_loss,
            'test_loss': test_loss,
            #'image_out': image_out,
            #'train_acc': acc,
            #'test_acc': acc,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        # Model is saved at the end of epoch
        self.model.save(self.sess)

    def train_step(self):
        # Next batch returns a generator object(iterator). Next function takes the next input from this object
        feed_dict = {self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.loss_function, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc

    def test_step(self):
        # Next batch returns a generator object(iterator). Next function takes the next input from this object
        feed_dict = {self.model.is_training: False}
        loss, acc = self.sess.run([self.model.loss_function, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc
