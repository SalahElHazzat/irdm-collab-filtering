import tensorflow.compat.v1 as tf
import keras.layers as rnn


# This class builds a Tensorflow graph which can be trained or used for inference
class MusicModel(object):

    def __init__(self, config):
        self._config = config
        # Create placeholders for the input and the targets
        self._input_data = tf.placeholder(tf.int32, [config.batch_size, config.num_steps])
        self._targets = tf.placeholder(tf.int32, [config.batch_size, config.num_steps])
        self._actual_seq_lengths = tf.placeholder(tf.int32, [config.batch_size])
        self._prediction = tf.placeholder(tf.int32, [config.batch_size, config.num_steps])
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size) # Create a basic LSTM cell
        # Now replicate the LSTM cell to create layers for a deep network
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

        # Make the initial state operator available
        self._initial_state = cell.zero_state(config.batch_size, tf.float32)

        # Map the inputs to their current embedding vectors
        # Embedding lookup must happen on the CPU as it is not currently supported on GPU
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [config.num_songs, config.embedding_size])
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, config.num_steps, inputs)]
        outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state,
                                 sequence_length=self._actual_seq_lengths)

        output = tf.reshape(tf.concat(1, outputs), [-1, config.hidden_size])
        softmax_w = tf.get_variable("softmax_w", [config.hidden_size, config.num_songs])
        softmax_b = tf.get_variable("softmax_b", [config.num_songs])
        logits = tf.matmul(output, softmax_w) + softmax_b

        # Compute the cross-entropy loss of the sequence by comparing each prediction with each target
        loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [tf.reshape(self._targets, [-1])],
                                                      [tf.ones([config.batch_size * config.num_steps])])
        # Added prediction
        self._prediction = logits

        # Expose the cost and final_state
        self._cost = tf.reduce_sum(loss) / config.batch_size
        self._final_state = state

    # Add training ops to the tensorflow graph and return the training operator
    def train_op(self):
        train_variables = tf.trainable_variables()  # variables to train
        # Implement gradient clipping to prevent gradient explosion
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, train_variables), self._config.max_grad_norm)

        # Using an Adam Optimiser
        optimizer = tf.train.AdamOptimizer()
        return optimizer.apply_gradients(zip(grads, train_variables))

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

    # Added preditions
    @property
    def prediction(self):
        return self._prediction

    @property
    def config(self):
        return self._config

    @property
    def actual_seq_lengths(self):
        return self._actual_seq_lengths

