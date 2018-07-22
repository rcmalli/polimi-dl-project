from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class DepthTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(DepthTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        # This function implements the operation in one epoch

        self.sess.run(self.data.iter_init_op)
        # Create an iterator
        loop = tqdm(range(self.config.num_iter_per_epoch))
        lose_list = []
        acc_list = []

        # Loop for the batch
        for _ in loop:
            loss, acc = self.train_step()
            lose_list.append(loss)
            acc_list.append(acc)

        loss = np.mean(lose_list)
        acc = np.mean(acc_list)

        # Epoch summary
        cur_it = self.model.global_step_tensor.eval(self.sess)

        # For observing additional parameters, it is enough to add it to summaries_dict with a proper tag
        summaries_dict = {
            'loss': loss,
            # 'acc': acc,
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
