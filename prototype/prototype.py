import numpy as np


class Prototype :

    def __init__(self ,embeddings):
        self.embeddings = embeddings

    def build_prototype(self, sample_embeddings):
        '''
        :param sample_embeddings : a list of document vectors; all documents belong to the same class
        :return: vector representation of the class; the prototype of the class
        '''
        return self.embeddings.centroid(sample_embeddings)


    def apply_prototype(self, instance_vecs, class_prototypes):
        '''
         :param instance_vecs: a list of instance vectors
         :param class_prototypes : a list of prototypes for each class
         :return: an array with the predictions for each instance (value 0 is the negative class, value 1 is the positive class)
         :return: score of the positive class 
         '''
        inst_scores = []
        for x in instance_vecs:
            scores = [ self.embeddings.vec_similarity(x,p) for p in class_prototypes ]
            inst_scores.append(scores)
        #print ("Instance scores")
        #print (np.array(inst_scores))
        probabilities = self.__softmax(np.array(inst_scores))
        #print ("probabilities")
        #print (probabilities)
        return np.argmax(probabilities, axis=1), probabilities[:,1]

    def debug_prototype(self, instances, rationale_vecs, class_prototypes):
        _, instance_weights = zip(*[self.embeddings.represent_bow(inst["text"].split(), bias_vecs=rationale_vecs, return_weights = True) for inst in instances])
        print ("review " + str(inst["rid"]))
        for (weights,inst) in zip(instance_weights, instances):
            word_weight_vec = sorted([ (weight, word) for (word, weight) in zip(inst["text"].split(), weights)])
            print('\n'.join(['%s\t%.4f' % (word, weight) for (weight,word) in word_weight_vec]))


    def __softmax(self, scores):
        '''
        :param scores : an array of shape instances x classes; instance scores for each class
        :return: an array of shape instances x classes; instance probabilities for each class
        '''
        x = scores - np.max(scores)
        return np.exp(x)/np.reshape(np.sum(np.exp(x),axis=1),(scores.shape[0],1))
        #return np.exp(x)/np.sum(np.exp(x))



