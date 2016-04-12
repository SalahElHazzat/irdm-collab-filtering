import tensorflow as tf


class Hook:
    def __init__(self, summary_writer, tag):
        self.summary_writer = summary_writer
        self.tag = tag

    def __call__(self, session, model, train_loss, epoch):
        raise NotImplementedError

    def update_summary(self, session, step, value):
        if self.summary_writer is not None:
            current_summary = tf.scalar_summary(self.tag, value)
            merged = tf.merge_summary([current_summary])
            summary_string = session.run(merged)
            self.summary_writer.add_summary(summary_string, step)


class TrainLossHook(Hook):
    def __init__(self, summary_writer, tag):
        super().__init__(summary_writer, tag)

    def __call__(self, session, model, train_loss, epoch):
        print("Epoch: %d Train Loss: %.3f" % (epoch, train_loss))
        self.update_summary(session, epoch, train_loss)


class GenericLossHook(Hook):
    def __init__(self, data, name, session_iterator, sequence_iterator, song_to_id, summary_writer, tag):
        super().__init__(summary_writer, tag)
        self._name = name
        self._data = data
        self._session_iterator = session_iterator
        self._sequence_iterator = sequence_iterator
        self._song_to_id = song_to_id

    # The implementation of this hook is very similar to training, except that train_op is not evaluated
    # because of course we're not trying to train the model, rather just evaluate it on some data set
    def __call__(self, session, model, train_loss, epoch):
        costs = 0.0

        for playlist in self._session_iterator(self._data, self._song_to_id, model.config.num_steps):
            state = model.initial_state.eval()
            for step, (x, y) in enumerate(self._sequence_iterator(playlist, model.config.num_steps)):
                cost, state = session.run([model.cost, model.final_state],
                                             {model.input_data: x,
                                              model.targets: y,
                                              model.initial_state: state})
                costs += cost

        print("Epoch: %d %s Loss: %.3f" % (epoch, self._name, costs))
        self.update_summary(session, epoch, costs)
