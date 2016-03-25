
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn
import lastfm_reader as data_reader


flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", None, "data_path")

FLAGS = flags.FLAGS


class MusicModel(object):

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size])
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, num_steps, inputs)]
        outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)

        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.nn.seq2seq.sequence_loss_by_example(
                [logits],
                [tf.reshape(self._targets, [-1])],
                [tf.ones([batch_size * num_steps])])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def train_op(self):
        return self._train_op


class Config(object):
    init_scale = 0.1
    max_grad_norm = 5
    num_layers = 2
    num_steps = 5
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    batch_size = 1


def run_epoch(session, m, data, song_to_id, eval_op):
    costs = 0.0
    iters = 0

    for playlist in data_reader.session_iterator(data, song_to_id):
        state = m.initial_state.eval()
        for step, (x, y) in enumerate(data_reader.seq_iterator(playlist, m.num_steps)):
            cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                         {m.input_data: x,
                                          m.targets: y,
                                          m.initial_state: state})
            costs += cost
            iters += m.num_steps

    return costs


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to data file")

    raw_data, songs = data_reader.read_data(FLAGS.data_path)
    # train_data, valid_data, test_data, _ = raw_data
    song_to_id = data_reader.get_song_to_id_map(songs)

    config = Config()
    eval_config = Config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = MusicModel(is_training=True, config=config)
        # with tf.variable_scope("model", reuse=True, initializer=initializer):
        #    mvalid = MusicModel(is_training=False, config=config)
        #    mtest = MusicModel(is_training=False, config=eval_config)

        tf.initialize_all_variables().run()

        for i in range(config.max_max_epoch):

            train_loss = run_epoch(session, m, raw_data, song_to_id, m.train_op)
            print("Epoch: %d Train Loss: %.3f" % (i + 1, train_loss))
            # valid_loss = run_epoch(session, mvalid, valid_data, tf.no_op())
            # print("Epoch: %d Valid Loss: %.3f" % (i + 1, valid_loss))

        # test_loss = run_epoch(session, mtest, test_data, tf.no_op())
        # print("Test Loss: %.3f" % test_loss)


if __name__ == "__main__":
    tf.app.run()
