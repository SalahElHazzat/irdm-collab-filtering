class Hook:
    def __init__(self):
        raise NotImplementedError

    def __call__(self, session, model, train_loss, epoch):
        raise NotImplementedError


class TrainLossHook(Hook):
    def __init__(self):
        pass

    def __call__(self, session, model, train_loss, epoch):
        print("Epoch: %d Train Loss: %.3f" % (epoch, train_loss))


class GenericLossHook(Hook):
    def __init__(self, data, name, session_iterator, sequence_iterator, song_to_id):
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


