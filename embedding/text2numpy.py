'''
Input: word embeddings text files, delimiter style
Output: an .npy and .vocab file that can be read efficiently with embeddings.py
'''
import numpy as np
import sys
import os


def space2underscore(text):
    return ''.join(['_' if c is ' ' else c for c in text])


def read_vectors_ws(path):
    '''
    Standard word2vec format
    '''
    vectors = {}
    with open(path) as input_f:
        for line in input_f:
            try:
                values = line.strip().split()
                word = values[0]
                if len(values) < 5:
                    sys.stderr.write(
                        "Skipping line: " + line)  # skipping header lines of empty line (assuming no embeddings with dim<5)
                else:
                    vectors[word] = np.asarray([float(x) for x in values[1:]])
            except ValueError as _:
                print(line.strip().split())
                sys.exit(1)
    return vectors


def read_vectors_tab(path):
    '''
    This format is used in Chang's embedding files
    '''
    vectors = {}
    with open(path) as input_f:
        for line in input_f:
            try:
                tokens = line.strip().split('\t')
                if len(tokens) < 2 or len(tokens[0].strip()) == 0:
                    sys.stderr.write(
                        "Skipping line: " + line)  # skipping header lines of empty line (assuming no embeddings with dim<10)
                    continue
                word = space2underscore(tokens[0])
                values = tokens[1].split()
                if len(values) < 10:
                    sys.stderr.write(
                        "Skipping line: " + line)  # skipping header lines of empty line (assuming no embeddings with dim<10)
                else:
                    vectors[word] = np.asarray([float(x) for x in values])
            except (ValueError, IndexError) as _:
                print(line.strip().split())
                sys.exit(1)
    return vectors


def read_vectors_comma(path):
    '''
    This format is used in Nazneen's embedding files
    '''
    vectors = {}
    with open(path) as input_f:
        for line in input_f:
            try:
                tokens = line.strip().split()
                word = tokens[0]
                values = tokens[1].split(',')
                if len(values) < 10:
                    sys.stderr.write(
                        "Skipping line: " + line)  # skipping header lines of empty line (assuming no embeddings with dim<10)
                else:
                    vectors[word] = np.asarray([float(x) for x in values])
            except ValueError as _:
                print(line.strip().split())
                sys.exit(1)
    return vectors


def text2numpy(inpath, outpath, delimiter):
    outpath_numpy = outpath + '.npy'
    if os.path.exists(outpath_numpy):
        sys.stderr.write(outpath + '.npy already exists. skipping.\n')
    else:
        if delimiter == 'ws':
            matrix = read_vectors_ws(inpath)
        elif delimiter == 'tab':
            matrix = read_vectors_tab(inpath)
        elif delimiter == 'comma':
            matrix = read_vectors_comma(inpath)
        else:
            print('Error: unknown delimiter type:', delimiter)
            return
        vocab = list(matrix.keys())
        vocab.sort()
        print('Vocab size: ' + str(len(vocab)))
        with open(outpath + '.vocab', 'w') as output_f:
            for word in vocab:
                output_f.write(word + '\n')

        new_matrix = np.zeros(shape=(len(vocab), len(matrix[vocab[0]])), dtype=np.float32)
        for i, word in enumerate(vocab):
            try:
                new_matrix[i, :] = matrix[word]
            except:
                print("Could not convert word: ", word)

        np.save(outpath_numpy, new_matrix)
        print('Numpy embeddings file saved to ' + outpath_numpy)


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Usage: %s <embedding-input-filename> <embedding-output-filename> [ws|tab|comma]" % sys.argv[0])
        sys.exit(1)

    inputf = sys.argv[1]
    outputf = sys.argv[2]
    if len(sys.argv) <= 3:
        delimiter = 'ws'
    else:
        delimiter = sys.argv[3]

    text2numpy(inputf, outputf, delimiter)


