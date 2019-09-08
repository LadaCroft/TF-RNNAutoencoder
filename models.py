import tensorflow as tf
from tensorflow.nn.rnn_cell import LSTMCell
from tensorflow.nn.rnn_cell import MultiRNNCell
from tensorflow.compat.v1.nn.rnn_cell import GRUCell
import numpy as np

# https://github.com/iwyoo/LSTM-autoencoder/blob/master/LSTMAutoencoder.py
# cf. http://arxiv.org/abs/1502.04681)


class RNNAutoencoder(object):

    def __init__(
        self,
        batch_size,
        inputs,
        outputs,
        num_units,
        cell_type
    ):
        """
    Args:
      inputs : a list (tensor array) of input tensors with size hp.num_time_steps*(batch_size,dim)
      cell : an rnn cell object (the default option is tf.python.ops.rnn_cell.LSTMCell)
      reverse : Option to decode in reverse order
      decode_without_input : Option to decode without input - there are zeros coming to the cell instead of input
    """

        self.batch_size = batch_size
        self.num_inputs = inputs[0].get_shape().as_list()[1]
        self.num_outputs = self.num_inputs

        num_hidden = num_units[-1]

        if len(num_units) > 1:
            if cell_type == 'GRU':
                cells = [GRUCell(num_units=n) for n in num_units]
            else:
                cells = [LSTMCell(num_units=n) for n in num_units]
            self._enc_cell = MultiRNNCell(cells)
            self._dec_cell = MultiRNNCell(cells)
        else:
            if cell_type == 'GRU':
                self._enc_cell = GRUCell(num_hidden)
                self._dec_cell = GRUCell(num_hidden)
            else:
                self._enc_cell = LSTMCell(num_hidden)
                self._dec_cell = LSTMCell(num_hidden)

        # , initializer=tf.contrib.layers.xavier_initializer()
        with tf.compat.v1.variable_scope('encoder') as es:
            enc_W = tf.Variable(tf.random.truncated_normal([num_hidden,
                                                            self.num_outputs], dtype=tf.float32), name='enc_weight'
                                )
            enc_b = tf.Variable(tf.random.truncated_normal([self.num_outputs],
                                                           dtype=tf.float32), name='enc_bias')

            init_states = []
            if cell_type == 'GRU':
                for i in range(len(num_units)):
                    layer = tf.zeros((batch_size, num_units[i]))
                    init_states.append(layer)
            else:
                # make the zero initial cell and hidden state as a tuple - in the shape LSTM cell expects it to be
                for i in range(len(num_units)):
                    init_c = tf.zeros((batch_size, num_units[i]))
                    init_h = init_c
                    layer = tf.contrib.rnn.LSTMStateTuple(init_c, init_h)
                    init_states.append(layer)
                init_states = tuple(init_states)

            if len(num_units) > 1:
                enc_state = init_states
            else:
                enc_state = init_states[0]

            enc_predictions = []
            for step in range(len(inputs)):
                if step > 0:
                    es.reuse_variables()
                enc_input = inputs[step]
                (enc_output, enc_state) = self._enc_cell(
                    enc_input, enc_state)  # lstm_output = hidden state, lstm_state = tuple(cell state, hidden state)
                #y_hat = Wy*h + by
                enc_prediction = tf.matmul(enc_output, enc_W) + enc_b
                enc_predictions.append(enc_prediction)

        with tf.compat.v1.variable_scope('decoder') as vs:
            dec_W = tf.Variable(tf.random.truncated_normal([num_hidden,
                                                            self.num_outputs], dtype=tf.float32), name='dec_weight'
                                )

            dec_b = tf.Variable(tf.random.truncated_normal([self.num_outputs],
                                                           dtype=tf.float32), name='dec_bias')

            dec_input = enc_prediction
            dec_state = enc_state
            dec_outputs = []
            for step in range(len(outputs)):
                if step > 0:
                    vs.reuse_variables()
                (dec_input, dec_state) = self._dec_cell(
                    dec_input, dec_state)
                dec_input = tf.matmul(dec_input, dec_W) + dec_b
                dec_outputs.append(dec_input)
            self.prediction = tf.transpose(
                tf.stack(dec_outputs), [1, 0, 2], name='prediction')

        self.input_ = tf.transpose(tf.stack(inputs), [1, 0, 2])
        self.target = tf.transpose(tf.stack(outputs), [1, 0, 2], name='target')
        self.prediction = self.prediction[:, :, 0]
        self.target = self.target[:, :, 0]
        self.enc_W = enc_W
        self.enc_b = enc_b
        self.dec_W = dec_W
        self.dec_b = dec_b


class RNNAugmented(object):

    def __init__(
        self,
        batch_size,
        inputs,
        outputs,
        num_units,
        cell_type
    ):
        """
    Args:
      num_hidden : number of hidden elements of each LSTM unit.
      inputs : a list (tensor array) of input tensors with size hp.num_time_steps*(batch_size,dim)
      cell : an rnn cell object (the default option is tf.python.ops.rnn_cell.LSTMCell)
      reverse : Option to decode in reverse order
      decode_without_input : Option to decode without input - there are zeros coming to the cell instead of input
    """

        self.batch_size = batch_size
        self.num_inputs = inputs[0].get_shape().as_list()[1]
        self.num_outputs = self.num_inputs
        num_time_steps = len(inputs)

        num_hidden = num_units[-1]
        self.last = inputs[-1]

        if len(num_units) > 1:
            cells = [LSTMCell(num_units=n) for n in num_units]
            self._lstm_cell = MultiRNNCell(cells)
        else:
            self._lstm_cell = LSTMCell(num_hidden)

        with tf.compat.v1.variable_scope('encoder') as ec:
            Wy = tf.Variable(tf.random.truncated_normal([num_hidden,
                                                         self.num_outputs], dtype=tf.float32), name='enc_weight'
                             )
            by = tf.Variable(tf.random.truncated_normal([self.num_outputs],
                                                        dtype=tf.float32), name='enc_bias')

            init_states = []
            for i in range(len(num_units)):
                init_c = tf.zeros((batch_size, num_units[i]))
                init_h = init_c
                layer = tf.contrib.rnn.LSTMStateTuple(init_c, init_h)
                init_states.append(layer)
            init_states = tuple(init_states)

            if len(num_units) > 1:
                lstm_state = init_states
            else:
                lstm_state = init_states[0]

            lstm_outputs = []
            for step in range(len(inputs)):
                if step > 0:
                    ec.reuse_variables()
                lstm_input = inputs[step]
                (lstm_output, lstm_state) = self._lstm_cell(
                    lstm_input, lstm_state)
            for step in range(len(outputs)):
                lstm_input = tf.matmul(lstm_output, Wy) + by
                lstm_outputs.append(lstm_input)
                (lstm_output, lstm_state) = self._lstm_cell(
                    lstm_input, lstm_state)

            self.prediction = tf.transpose(
                tf.stack(lstm_outputs), [1, 0, 2], name='prediction')
            self.target = tf.transpose(
                tf.stack(outputs), [1, 0, 2], name='target')
            self.input_ = tf.transpose(tf.stack(inputs), [1, 0, 2])
            self.prediction = self.prediction[:, :, 0]
            self.target = self.target[:, :, 0]
            self.enc_W = Wy
            self.enc_b = by


def init_model(model_type, current_batch_size, encoder_inputs_ta, decoder_targets_ta, num_units, cell_type):
    if model_type == 'RNNAutoencoder':
        model = RNNAutoencoder(current_batch_size,
                               encoder_inputs_ta, decoder_targets_ta, num_units, cell_type)
    elif model_type == 'RNNAugmented':
        model = RNNAugmented(current_batch_size,
                             encoder_inputs_ta, decoder_targets_ta, num_units, cell_type)
    else:
        raise Exception('Model type was not recognized.')

    return model
