import tensorflow as tf
import numpy as np
import hyperparameters as hp


class Optimize(object):

    def __init__(
        self,
        loss,
        gradient_clipping=False
    ):
        """
        Args:
        loss : current loss in iteration during training
        gradient_clipping : condition whether we want to use gradient clipping
        """

        self.name = hp.optimizer_type
        self.loss = loss

        with tf.name_scope('train'):
            if hp.optimizer_type == 'Adam':
                optimizer = tf.compat.v1.train.AdamOptimizer(
                    learning_rate=hp.learning_rate, name=self.name)
            elif hp.optimizer_type == 'RMSProp':
                optimizer = tf.train.RMSPropOptimizer(
                    learning_rate=hp.learning_rate, name=self.name)
            elif hp.optimizer_type == 'GD':
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=hp.learning_rate, name=self.name)

            if hp.optimizer_type is None:
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=hp.learning_rate, name='Adam')

            self.optimizer = optimizer

            if gradient_clipping == True:
                # Gradient Clipping
                # https://stackoverflow.com/questions/36498127/how-to-apply-gradient-clipping-in-tensorflow
                gradients, variables = zip(*optimizer.compute_gradients(loss))
                gradients = [
                    None if gradient is None else tf.clip_by_norm(
                        gradient, 5.0)  # L2norm is <= 5
                    for gradient in gradients]
                training_op = optimizer.apply_gradients(
                    zip(gradients, variables))
            else:
                training_op = optimizer.minimize(self.loss, name='training')

            self.training_op = training_op
