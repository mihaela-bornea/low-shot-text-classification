'''
Efficient representation of word embeddings
'''

import numpy as np
import heapq
import os
from scipy.sparse import dok_matrix, csr_matrix

class Embeddings:

    def __init__(self, path, unk='<unk>', normalize=True, one_hot=False, stats_count=False):
        '''
        :param path: path where embeddings file .npy and .vocab are found
        :param unk: the special unknown word key. If unk is not in the vocab then an all-zeros vector for unknown words.
        :param normalize: whether to L2 normalize the embedding vectors to 1.0
        :param one_hot: represent words as 'one-hot' vectors (ignores embeddings representation)
        '''

        self.one_hot = one_hot
        self.normalize_flag = normalize
        self.unk = unk
        if path is not None:
            if not one_hot:
                if os.path.exists(path + '.npy'):
                    self.m = np.load(path + '.npy')
                    self.wi, self.iw = self._read_vocab(path + '.vocab')
                    self.dim = self.m.shape[1]
                    if unk not in self.wi:
                        unk_vec = np.zeros((1,self.dim), dtype=np.float32)
                        self.m = np.concatenate((self.m,unk_vec))
                        self.wi[unk] = len(self.wi)
                        self.iw.append(unk)
                    if normalize:
                        self.m = self.__normalize(self.m)
                else:
                    raise Exception("Embeddings file does not exist: %s.npy" % path)
            else: # one_hot
                self.wi, self.iw = self._read_vocab(path + '.vocab')
                if unk not in self.wi:
                    self.wi[unk] = len(self.wi)
                    self.iw.append(unk)

                self.dim = len(self.iw)
                self.m = dok_matrix((self.dim, self.dim), dtype=np.float32)
                for i in range(self.dim):
                    self.m[i,i] = 1.0
                self.m = self.m.tocsr()

        #debug stats
        self.stats_count = stats_count
        if self.stats_count:
            self.unks = 0
            self.total_toks = 0

    def get_unk_ratio(self):
        return float(self.unks)/self.total_toks

    def is_context_sensitive(self):
        return False

    def is_seq_embedder(self):
        return False

    def is_one_hot(self):
        return self.one_hot

    def size(self):
        return self.m.shape[0]


    def units(self):
        return self.dim


    def __contains__(self, w):
        if (isinstance(w, list)):
            return len([ww for ww in w if ww in self.wi]) == len(w)
        else:
            return w in self.wi


    def represent(self, w):
        '''
        :param w: a word
        :return: vector representation of the word
        '''
        return self.m[self.wi[w], :] if w in self.wi else self.m[self.wi[self.unk], :]


    def represent_text(self, text):
        return self.represent_bow(text)

    def represent_bow(self, bow):
        '''
        :param bow: list of words
        :return matrix with a vector for each word
        '''

        bow_indices = [self.wi[word] if word in self.wi else self.wi[self.unk] for word in bow]
        bow_mat = self.m[bow_indices, :]

        if self.stats_count:
            self.unks += sum((1 if len(word)>1 and word not in self.wi else 0 for word in bow)) # len(word)==1 means mostly punctuation
            self.total_toks += len(bow_indices)

        # bow_inds = [self.m[self.wi[word],:] if word in self.wi else self.m[self.wi[self.unk],:] for word in bow]
        # bow_mat = np.stack(bow_vecs)

        # if isinstance(bow_mat, csr_matrix):
        #     bow_mat = bow_mat.toarray()

        return bow_mat

    def closest(self, w, n=10):
        '''
        :param w: a word
        :return the top n words closest to w (if w is a list of words then the bow centroid is used)
        '''
        return self.closest_to_vec(self.represent(w), n)


    def closest_to_vec(self, vec, n=10):
        # scores = ((self.m.dot(vec) + 1.0) / 2)
        scores = self.m.dot(vec)
        return heapq.nlargest(n, zip(scores, self.iw))

    def similarity(self, w1, w2):
        '''
         :param w1: word 1
         :param w2: word 2
         :return similarity between w1 and w2. If self.normalize==True then this is cosine similarity
         '''
        return self.vec_similarity(self.represent(w1), self.represent(w2))


    def vec_similarity(self, v1, v2):
        # return (v1.dot(v2) + 1.0) / 2
        return v1.dot(v2)


    def __normalize_vec(self, v):
        v = v.squeeze()
        norm = np.sqrt(np.dot(v, v))
        return v / norm if norm != 0 else v


    def __normalize(self, m):
        norm = np.sqrt(np.sum(m * m, axis=1))
        norm[norm == 0.0] = 1.0
        return m / norm[:, np.newaxis]

    def _read_vocab(self, path):
        vocab = []
        with open(path) as f:
            for line in f:
                vocab.extend(line.strip().split())
        return dict([(w, i) for i, w in enumerate(vocab)]), vocab






