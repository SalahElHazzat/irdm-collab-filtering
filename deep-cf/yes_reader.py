import collections
import numpy as np
import math


def get_data(filepath):
    with open(filepath) as f:
        lines = f.read().splitlines()

    data = [[int(x) for x in line.split()] for line in lines[2::] if len(line.split()) > 1]
    songs = set([d for sublist in [v for v in data] for d in sublist])

    total = len(data)
    train_start = 0
    train_end = int(math.floor(total * 0.6))
    validation_start = int(train_end + 1)
    validation_end = int(total)

    train_data = data[train_start:train_end]
    validation_data = data[validation_start:validation_end]
    return train_data, validation_data, None, songs


def get_test_data(filepath):
    with open(filepath) as f:
        lines = f.read().splitlines()

    test_data = [[int(x) for x in line.split()] for line in lines[2::] if len(line.split()) > 1]
    songs = set([d for sublist in [v for v in test_data] for d in sublist])

    # songs already given to us in the data files
    return test_data, songs


def get_song_to_id_map(songs):
    counter = collections.Counter(songs)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    songs, _ = list(zip(*count_pairs))
    song_to_id = dict(zip(songs, range(len(songs))))

    return song_to_id


def session_iterator(data, song_to_id, _, batch_size):
    num_batches = len(data) // batch_size
    for batch in range(0, num_batches):
        yield session_to_seq(data[batch*batch_size:(batch+1)*batch_size], song_to_id)


def session_to_seq(session, song_to_id):
    return np.array([[song_to_id[song] for song in sess] for sess in session])


def seq_iterator(sequence, seq_length):
    batch_size = len(sequence)
    n = int(math.ceil(max([len(seq) for seq in sequence]) / seq_length))
    for i in range(n):  # Iterate over the distinct sequences
        x_arr = np.zeros([batch_size, seq_length])
        y_arr = np.zeros([batch_size, seq_length])
        actual_lengths = []

        for j in range(0, batch_size):  # Iterate over each row in the batch
            x = sequence[j][i * seq_length:(i + 1) * seq_length]
            y = sequence[j][i * seq_length + 1:(i + 1) * seq_length + 1]
            x_arr[j, 0:len(x)] = x
            y_arr[j, 0:len(y)] = y
            actual_lengths.append(len(x))

        yield (x_arr, y_arr, np.array(actual_lengths))


def run():
    filepath = '../../data/MIT-dataset/yes_big/train.txt'
    data = get_data(filepath)
    print(data)

if __name__ == "__main__":
    run()

