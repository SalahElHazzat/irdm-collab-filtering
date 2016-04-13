import collections
import numpy as np
import math


def get_data(filepath):
    (training_start, training_end), (validation_start, validation_end), (test_start, test_end) = data_splits()
    raw_data = read_data(filepath, 100000)

    training_set, train_songs = group_data_subset(raw_data, training_start, training_end)
    validation_set, validation_songs = group_data_subset(raw_data, validation_start, validation_end)
    test_set, test_songs = group_data_subset(raw_data, test_start, test_end)
    return training_set, validation_set, test_set, train_songs.union(validation_songs).union(test_songs)


def read_data(filepath, limit):
    with open(filepath) as f:
        raw_data = []

        for i in range(0, limit):
            line = f.readline().rstrip('\n')
            components = line.split('\t')
            user_id = components[0]
            # timestamp = components[1]
            # song = (components[3], components[5])
            song = (components[3])
            raw_data.append((user_id, song))
        return raw_data


def group_data_subset(raw_data, start, end):
    data = {}
    for user_id, song in raw_data[start:end]:
        if user_id not in data:
            data[user_id] = []
        data[user_id].append(song)

    songs = set([song for sublist in [v for _, v in data.items()] for song in sublist])
    return data, songs


def data_splits():
    # parameters
    # total_obs = 19098862
    total_obs = 100000
    total_obs_considered = 100000  # -->equal or lower than total_obs
    training_ratio = 0.6
    validation_ratio = 0.2
    test_ratio = 0.2

    # setting number of obs in each set
    training_q = math.floor(total_obs_considered * training_ratio)
    validation_q = math.floor(total_obs_considered * validation_ratio)
    test_q = total_obs_considered - training_q - validation_q

    # training_start = math.floor(total_obs * np.random.uniform(low=0.0, high=1.0) - 1)
    training_start = 0
    training_end = training_start + training_q - 1

    if training_end > total_obs - 1:
        training_end -= total_obs

    validation_start = training_end + 1

    if validation_start > total_obs - 1:
        validation_start -= total_obs

    validation_end = validation_start + validation_q - 1

    if validation_end > total_obs - 1:
        validation_end -= total_obs

    test_start = validation_end + 1

    if test_start > total_obs - 1:
        test_start -= total_obs

    test_end = test_start + test_q - 1

    if test_end > total_obs - 1:
        test_end -= total_obs

    return (training_start, training_end), (validation_start, validation_end), (test_start, test_end)


def get_song_to_id_map(songs):
    counter = collections.Counter(songs)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    songs, _ = list(zip(*count_pairs))
    song_to_id = dict(zip(songs, range(len(songs))))

    return song_to_id


def session_iterator(data, song_to_id, seq_length):
    for key in data:
        if len(data[key]) <= seq_length:
            print("Warning: skipping playlist due to length")
            continue

        yield session_to_seq(data[key], song_to_id)


def seq_iterator(sequence, seq_length):
    n = (len(sequence) - 1) // seq_length
    for i in range(n):
        x = sequence[i * seq_length:(i + 1) * seq_length]
        y = sequence[i * seq_length + 1:(i + 1) * seq_length + 1]
        yield (np.array([x]), np.array([y]))


def session_to_seq(session, song_to_id):
    return np.array([song_to_id[song] for song in session])


def run():
    filepath = '../../data/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv'
    data, songs = read_data(filepath)
    song_to_id = get_song_to_id_map(songs)
    for session in session_iterator(data, song_to_id):
        for seq in seq_iterator(session, 2):
            print(seq)


if __name__ == "__main__":
    run()
