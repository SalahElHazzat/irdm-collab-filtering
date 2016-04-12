import tensorflow as tf


class Trainer:

    def __init__(self, model, data, song_to_id, session_iterator, sequence_iterator):
        self._model = model
        self._num_steps = model.config.num_steps
        self._data = data
        self._song_to_id = song_to_id
        self._train_op = model.train_op()
        self._session_iterator = session_iterator
        self._sequence_iterator = sequence_iterator

    def train(self, session, num_epochs, hooks):
        tf.initialize_all_variables().run()
        for i in range(0, num_epochs):
            train_loss = self.run_epoch(session)
            for hook in hooks:
                hook(session, self._model, train_loss, i+1)

    def run_epoch(self, session):
        costs = 0.0
        # iters = 0

        for playlist in self._session_iterator(self._data, self._song_to_id, self._num_steps):
            state = self._model.initial_state.eval()  # Initialise the state

            # Despite having a playlist of length N, we can only pass num_steps songs to the RNN at a time,
            # corresponding to the number of cells that have been "unravelled". To handle this, we split the
            # playlist into num_steps chunks, run it through the unravelled RNN, keep track of the final state
            # and pass that back to the RNN as its initial state when running for the next sequence.
            for step, (x, y) in enumerate(self._sequence_iterator(playlist, self._num_steps)):

                # Now the key part of training - set the input data to x, the targets to y and the initial state to
                # the current state. Ask tensorflow to evaluate the model_cost, final_state and train_op operators in
                # the computation graph. The results of model_cost and state are used to update the current cumulative
                # cost, while evaluation of train_op results in backpropagation of any prediction errors into the
                # network to update the network's parameters (in particular the embeddings and the logistic regression
                # parameters)
                cost, state, _ = session.run([self._model.cost, self._model.final_state, self._train_op],
                                             {self._model.input_data: x,
                                              self._model.targets: y,
                                              self._model.initial_state: state})
                costs += cost
                # iters += self._num_steps

        return costs

