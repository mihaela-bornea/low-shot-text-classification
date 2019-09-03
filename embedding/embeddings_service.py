
import numpy as np
import itertools
from utils.arithmetic_utils import *


class TextSpan:

    
    def __init__(self, text, begin=0, end=-1, weights=None, sow=False):
        '''
        [begin:end] is the `span' and the text around [begin:end] is considered the `context' of the span
        :param text: list of words
        :param begin:
        :param end:
        :param weights: optional weights assigned to every token in text
        :param sow: set-of-words representation
        '''

        assert(type(text) == list)
        self.begin = begin
        self.end = end if end > 0 else len(text)
        self.text = text
        self.weights = weights
        assert ((self.begin >= 0) and (self.end > self.begin) and (self.end <= len(self.text)))
        assert(self.weights is None or len(self.text) == len(self.weights))
        if sow:
            self.to_set_of_words()

    def to_set_of_words(self):
        '''
        removes duplicate words (used in set-of-words representations). Note: original order of words is not preserved.
        :return:
        '''
        assert(self.begin==0 and self.end==len(self.text))

        span_words_set = set(self.text[self.begin:self.end])
        new_text = list(span_words_set)
        if self.weights is not None:
            word2weight = {}
            for word, weight in zip(self.text, self.weights):
                if word not in word2weight or (word in word2weight and weight > word2weight[word]):
                    word2weight[word] = weight
            self.weights = [word2weight[word] for word in new_text]
 
        # context_words_set = set(self.text[:self.begin] + self.text[self.end:]) - span_words_set
        # self.text = list(span_words_set) + list(context_words_set)
        self.text = new_text
        self.begin = 0
        self.end = len(self.text)
        return self

    def mult_weights(self, word2weight, default_weight=1.0):
        if self.weights == None:
            self.weights = [1.0]*len(self.text)
        for i, word in enumerate(self.text):
            self.weights[i] = self.weights[i]*word2weight[word] if word in word2weight else self.weights[i]*default_weight

    def get_span_words(self):
        return self.text[self.begin:self.end]

    def get_span_weights(self):
        return self.weights[self.begin:self.end] if self.weights is not None else None

    def is_full(self):
        return self.end-self.begin == len(self.text)

    def __str__(self):
        weights_str = "" if self.weights is None else ' '.join([str(w) for w in self.weights])
        return '%d %d %s w = [%s]' % (self.begin, self.end, self.text, weights_str)

class EmbeddingsService:

    def __init__(self, embeddings, normalize=False, bias_strength=1.0, stopwords={}, sow_representation=False, idfs=None):
        '''
        :param embeddings: embedding object
        :param normalize: whether to L2 normalize all returned embeddings to 1.0
        :param bias_strength: determines how strong the bias effect is when constructing biased bow representations (0 - no effect).
        :param stopwords: a set of words to be ignored
        :param sow_representation: consider text input as a set-of-words as opposed to bag-of-words, i.e. every word type is only count once 
        (doen't work for context sensitive embeddings)
        '''

        assert(not (sow_representation and (embeddings.is_context_sensitive() or embeddings.is_seq_embedder())))
        self.embeddings = embeddings

        if (embeddings.is_seq_embedder):
            print('NOTICE: Using a sequence embedder. Entire sequences (sentences) will be embedded as whole disregarding span boundaries within them.')

        # if binary == True:
            # assert (normalize == False and embeddings.is_one_hot() == True and bias_strength == 0.0)

        self.normalize_flag = normalize
        self.sow = sow_representation
        self.bias_strength = bias_strength
        self.stopwords = stopwords
        self.idfs = idfs
    
    def set_idfs(self, idfs):
        self.idfs = idfs
        
    def embeddings_size(self):
        return self.embeddings.size()

    def embeddings_units(self):
        return self.embeddings.units()

    def __contains__(self, w):
        return w in self.embeddings


    def represent(self, w):
        '''
        :param w: a word
        :return: vector representation of the word
        '''
        return self.embeddings.represent(w)


    def represent_spans(self, spans, use_words, bias_vecs=None, return_weights=False):
        '''
        :param spans: list of text spans to be represented
        :param bias_vecs: vecs used to bias the relative importance weight of each word across all the input spans
        :param return_weights: whether to return the final weights used (mostly for debug)
        :param use_words: vocabulary words to use, OOV is ignored (None means no restriction)
        :return: A single weighted average vector representation for all the spans concatenated; optional - the weights used
        '''

        if len(spans) == 0:
            unk_span = TextSpan(['<unknownword>'])
            spans = [unk_span]

        if self.idfs != None:
            for s in spans:
                s.mult_weights(self.idfs)

        if self.embeddings.is_seq_embedder():

            text_matrix = self.embeddings.represent_text_batch([span.text for span in spans])
            span_weights = [1]*len(spans)
            text_weights = np.reshape(np.asarray(span_weights), (len(span_weights), 1))

        elif self.embeddings.is_context_sensitive() :
            span_matrices, spans_weights = zip(*[self.embed_span(span, use_words) for span in spans])

            if len(span_matrices) > 1:
                text_matrix = mat_concat(span_matrices)
                text_weights = mat_concat(spans_weights)
            else:
                text_matrix = span_matrices[0]
                text_weights = spans_weights[0]
        else:
            concatenated_span_words = list(itertools.chain.from_iterable([s.get_span_words() for s in spans]))
            concatenated_span_weights = list(itertools.chain.from_iterable([s.get_span_weights() if s.get_span_weights() is not None else [] for s in spans]))
            if len(concatenated_span_weights) == 0:
                concatenated_span_weights = None
            concatenated_span = TextSpan(concatenated_span_words, weights=concatenated_span_weights, sow=self.sow)
            text_matrix, text_weights = self.embed_span(concatenated_span, use_words)

        if bias_vecs is None or self.bias_strength == 0:
            bias_weights = np.ones(text_weights.shape)
        else:
            bias_weights = self._apply_bias_metric(text_matrix, bias_vecs)
            bias_weights = np.reshape(bias_weights, (bias_weights.size, 1))

        final_weights = pointwise_mult(bias_weights, text_weights)

        if final_weights.sum() != 0:
            final_weights = final_weights / final_weights.sum()

        weighted_text_mat = pointwise_mult(text_matrix, final_weights)
        avg_text = mat_sum(weighted_text_mat, axis=0)

        # if self.binary_flag:
        #     avg_text[np.nonzero(avg_text)] = 1.0

        if self.normalize_flag:
            avg_text = self._normalize_vec(avg_text)

        if return_weights:
            return avg_text, final_weights.squeeze().tolist()
        else:
            return avg_text

    def represent_span(self, use_words, text, begin=0, end=-1, bias_vecs=None, return_weights=False, weights=None):
        '''
        A convenience method to use when there's only one span to be represented
        '''
        spans = [TextSpan(text, begin, end, weights)]
        return self.represent_spans(spans, use_words, bias_vecs, return_weights)

    def embed_span(self, span, use_words):
        '''
        :param span: input text span
        :param use_words: vocabulary words to use, OOV is ignored (None means no restriction)
        :return: matrix of words x dim, list of weights per word (zero weights for ignored words)
        '''

        text_matrix = self.embeddings.represent_text(span.text)

    # if not self.embeddings.is_seq_embedder:

        span_words = span.text[span.begin:span.end]

        weights = span.weights if span.weights is not None else [1]*(span.end-span.begin)
        
        span_weights = [weights[i - span.begin] if span_words[i - span.begin].lower() not in self.stopwords and
                (use_words == None or span_words[i - span.begin] in use_words) and
                                                span_words[i - span.begin] in self.embeddings
                        else 0.0 for i in range(span.begin, span.end)]

        # weights = [self.span_context_weight] * (span.begin) + [1.0] * (span.end - span.begin) + [
        #     self.span_context_weight] * (len(span.text) - span.end)

        # span_weights = [weights[i] if span.text[i].lower() not in self.stopwords and
        #                                            (use_words == None or span.text[i] in use_words) and
        #                          span.text[i] in self.embeddings
        #                 else 0.0 for i in range(len(span.text))]

        if not span.is_full():
            span_matrix = text_matrix[span.begin:span.end, :]
            # span_weights = span_weights[span.begin:span.end]
        else:
            span_matrix = text_matrix

    # else:
    #     span_matrix = text_matrix
    #     span_weights = [1]

        span_weights = np.reshape(np.asarray(span_weights), (len(span_weights), 1))

        # print(span.begin, span.end, span_matrix.shape)
        return (span_matrix, span_weights)


    def _apply_bias_metric(self, target_vecs, bias_vecs):
        '''

        :param target_vecs: vectors of words
        :param bias_vec: the point bow is to be biased towards
        :return: vector of bias weights (per each word in the input)
        '''

        # bias_vecs = np.stack(bias_vecs)
        bias_vecs = mat_concat(bias_vecs)

        target_norms = compute_norm(target_vecs)
        bias_norms = compute_norm(bias_vecs)
        norms = matmul(target_norms, bias_norms.T)
        bias_weights = matmul(target_vecs, bias_vecs.T) / norms

        bias_weights = np.max(bias_weights, axis=1)
        bias_weights = np.exp(bias_weights - np.max(bias_weights))
        bias_weights = np.power(bias_weights, self.bias_strength)
        return np.asarray(bias_weights)

    # def get_idfs(self, vecs):
    #     '''
    #     :param vecs: list of vectors, each representing a text 'doc'
    #     :return: idf score vector
    #     '''
    #     mat = np.asarray(vecs)
    #     counts = np.sum(mat, axis=0)
    #     idfs = np.log( float(1 + len(vecs)) / (1 + counts) ) + 1
    #     return idfs
        
    #     # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
    #     # idf(d, t) = log [ (1 + n) / (1 + df(d, t)) ] + 1

    # def set_idfs(self, idfs):
    #     assert(self.embeddings.is_one_hot())
    #     self.idfs = idfs

    def centroid(self, vecs):
        '''

        :param vecs: list of vectors
        :return: centroid of the vectors
        '''
        mat = np.asarray(vecs)
        centroid = np.average(mat, axis=0)

        # if self.binary_flag:
        #     centroid[np.nonzero(centroid)] = 1.0

        if self.normalize_flag:
            centroid = self._normalize_vec(centroid)
        return centroid


    def similarity(self, w1, w2):
        return self.vec_similarity(self.represent(w1), self.represent(w2))


    def vec_similarity(self, v1, v2):
        # return (v1.dot(v2) + 1.0) / 2
        return self._normalize_vec(v1.squeeze()).dot(self._normalize_vec(v2.squeeze()))


    def add_vecs(self, v1, v2):
        v3 = v1+v2
        if self.normalize_flag:
            v3 = self._normalize_vec(v3)
        return v3


    def _normalize_vec(self, v):

        shape = v.shape
        v = v.squeeze()
        norm = np.sqrt(np.dot(v, v))
        v = v / norm if norm != 0 else v
        v = np.reshape(v, shape)
        return v


    def _normalize(self, m):
        norm = np.sqrt(np.sum(m * m, axis=1))
        norm[norm == 0.0] = 1.0
        return m / norm[:, np.newaxis]



