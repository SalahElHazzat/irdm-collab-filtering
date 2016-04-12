from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import lastfm_reader as data_reader
from musicmodel import MusicModel
from modeltrainer import Trainer
from hooks import *


flags = tf.flags
logging = tf.logging

# Flags allows for processing of command-line arguments
flags.DEFINE_string("data_path", None, "data_path")
FLAGS = flags.FLAGS


class ModelConfig(object):
    init_scale = 0.1
    max_grad_norm = 5
    num_layers = 2
    num_steps = 10
    hidden_size = 200
    embedding_size = 200
    num_epochs = 20
    batch_size = 1


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to data file")

    train_data, valid_data, test_data, songs = data_reader.get_data(FLAGS.data_path)
    song_to_id = data_reader.get_song_to_id_map(songs)

    train_config = ModelConfig()
    train_config.num_songs = len(songs)

    eval_config = ModelConfig()
    eval_config.batch_size = 1
    eval_config.num_steps = 1
    eval_config.num_songs = len(songs)

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-train_config.init_scale,
                                                    train_config.init_scale)

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = MusicModel(train_config)

        hooks = [
            TrainLossHook(),
            GenericLossHook(valid_data, 'Validation', data_reader.session_iterator, data_reader.seq_iterator, song_to_id),
        ]

        trainer = Trainer(m, train_data, song_to_id, data_reader.session_iterator, data_reader.seq_iterator)
        trainer.train(session, train_config.num_epochs, hooks)


if __name__ == "__main__":
    tf.app.run()
