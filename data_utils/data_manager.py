import json
import numpy as np


class DataManager :

    def __init__(self, train_file, dev_file, test_file, aspect, dev = True, sent_split = True):

        self.random_gen = np.random.RandomState(2222)
        self.evaluate_dev = dev
        self.sent_split = sent_split
        print("Initialize data manager ")

        print("Loading training data from " + train_file)
        with open(train_file,"r") as read_train :
            train_instances = json.load(read_train)
            self.pos_train_instances = []
            self.neg_train_instances = []
            for t_i in train_instances:
                aspect_t_i = self.__extract_aspect_instance(t_i, aspect)
                if len(aspect_t_i["text"]) == 0 :
                    continue
                if aspect_t_i["label"] == 0 :
                    self.neg_train_instances.append(aspect_t_i)
                elif  aspect_t_i["label"] == 1 :
                    self.pos_train_instances.append(aspect_t_i)
            self.train_instances = []
            self.train_instances.extend(self.pos_train_instances)
            self.train_instances.extend(self.neg_train_instances)

        print ("Training size " +str(len(self.pos_train_instances)+len(self.neg_train_instances)) +
               " positives " + str(len(self.pos_train_instances)) +
               " negatives " + str(len(self.neg_train_instances)))


        if dev:
            pos_dev = 0
            neg_dev = 0
            self.dev_instances = []

            print("Loading dev data from " + dev_file)
            
            with open(dev_file,"r") as read_dev :
                dev_instances = json.load(read_dev)
                for d_i in dev_instances:
                    aspect_d_i = self.__extract_aspect_instance(d_i, aspect)

                    if len(aspect_d_i["text"]) == 0 :
                        continue
                    if aspect_d_i["label"] == 1 :
                        pos_dev = pos_dev + 1
                        self.dev_instances.append(aspect_d_i)
                    elif aspect_d_i["label"] == 0:
                        neg_dev = neg_dev + 1
                        self.dev_instances.append(aspect_d_i)

            print ("Dev size  " + str(len(self.dev_instances)) +
               " positives " + str(pos_dev) +
               " negatives " + str(neg_dev))


        self.aspect = aspect

        if not dev and test_file != None:
            pos_test = 0
            neg_test = 0
            print("Loading test data from " + test_file)
            self.test_instances = []
            with open(test_file,"r") as read_test :
                test_instances = json.load(read_test)
                for t_i in test_instances:
                    aspect_t_i = self.__extract_aspect_instance(t_i, aspect)
                    if len(aspect_t_i["text"]) == 0 :
                        continue
                    if aspect_t_i["label"] == 1 :
                        pos_test = pos_test + 1
                        self.test_instances.append(aspect_t_i)
                    elif aspect_t_i["label"] == 0 :
                        neg_test = neg_test + 1
                        self.test_instances.append(aspect_t_i)
            print ("Aspect(task) of interest is aspect "+str(aspect))

            print("Test size " + str(len(self.test_instances)) +
                " positives " + str(pos_test) +
                " negatives " + str(neg_test))


    def get_data(self, n=-1) :
        '''
        :return the dev instances
        '''
        if self.evaluate_dev :
            data = self.dev_instances
        else :
            data = self.test_instances
        return data if n<=0 else data[:n]

    def get_pos_train_sample(self, sample_size):
        '''
        :param sample_size: the size of the train sample
        :return a sample of the positive training instances
        '''
        if sample_size > 0:
            return [ self.pos_train_instances[i] for i in self.random_gen.randint(len(self.pos_train_instances), size = sample_size) ]
        else:
            full_sample = self.pos_train_instances.copy()
            self.random_gen.shuffle(full_sample)
            return full_sample

    def get_neg_train_sample(self, sample_size):
        '''
        :param sample_size: the size of the train sample
        :return a sample of the negative training instances
        '''
        
        if sample_size > 0:
            return [ self.neg_train_instances[i] for i in self.random_gen.randint(len(self.neg_train_instances), size = sample_size) ]
        else:
            full_sample = self.neg_train_instances.copy()
            self.random_gen.shuffle(full_sample)
            return full_sample

    def get_train_sample(self, sample_size):
        '''
        :param sample_size: the size of the train sample
        :return a sample of the training instances
        '''
        if sample_size > 0:
            return [ self.train_instances[i] for i in self.random_gen.randint(len(self.train_instances), size = sample_size) ]
        else:
            full_sample = self.train_instances.copy()
            self.random_gen.shuffle(full_sample)
            return full_sample

    def get_train_data(self):
        '''
        :return the train instances
        '''
        train_data = []
        train_data.extend(self.neg_train_instances)
        train_data.extend(self.pos_train_instances)
        return train_data

    def __extract_aspect_instance(self,inst,aspect):
        aspect_inst = {}
        aspect_inst["rid"] = inst["rid"]
        aspect_inst["text"] = inst["text"]
        if self.sent_split:
            aspect_inst["sents"] = inst["sents"]
        else:
            aspect_inst["sents"] =[[0,(len(inst["text"])-1)]]
        label_string = "label-"+str(aspect)
        spans_string = "spans-"+str(aspect)
        if float(inst[label_string]) > 0.8 :
            aspect_inst["label"] = 1
        elif float(inst[label_string]) < 0.7 :
            aspect_inst["label"] = 0
        else :
            aspect_inst["label"] = None
        aspect_inst["spans"]=inst[spans_string]
        return aspect_inst
