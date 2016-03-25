import collections
import numpy as np
from itertools import groupby


def read_data(filepath):
    with open(filepath) as f:
        raw_data = []

        for i in range(0, 10000):
            line = f.readline().rstrip('\n')
            components = line.split('\t')
            user_id = components[0]
            # timestamp = components[1]
            song = (components[3], components[5])

            raw_data.append((user_id, song))

        songs = [song for _, song in raw_data]
        return groupby(raw_data, lambda d: d[0]), songs


def get_song_to_id_map(songs):
    counter = collections.Counter(songs)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    songs, _ = list(zip(*count_pairs))
    song_to_id = dict(zip(songs, range(len(songs))))

    return song_to_id


def session_iterator(data, song_to_id):
    for session in data:
        yield session_to_seq(session, song_to_id)


def seq_iterator(sequence, seq_length):
    n = len(sequence) - 1 // seq_length
    for i in range(n):
        x = sequence[i*seq_length:(i+1)*seq_length]
        y = sequence[i*seq_length+1:(i+1)*seq_length+1]
        yield (np.array([x]), np.array([y]))


def session_to_seq(session, song_to_id):
    _, songs = session
    return np.array([song_to_id[song] for _, song in songs])


def run():
    filepath = '/Volumes/SAMSUNG/Studying/irdm-project/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv'
    data, songs = read_data(filepath)
    song_to_id = get_song_to_id_map(songs)
    for session in session_iterator(data, song_to_id):
        for seq in seq_iterator(session, 2):
            print(seq)


if __name__ == "__main__":
    run()

