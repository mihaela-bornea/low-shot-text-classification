import sys

import torch
from fastai.gen_doc.nbdoc import *
from fastai import *        # Quick accesss to most common functionality
from fastai.text import *   # Quick accesss to NLP functionality

import pandas as pd
import numpy as np
import shutil
import os

#You should lower the batch size or the bptt



class UlmfitClassifier(object):


    def __init__(self, path):
        self.train_size = 0
        self.valid_size = 0

        # S3 = 'https://s3.amazonaws.com/fast-ai-'
        # S3_NLP = f'{S3}nlp/'
        # IMDB = f'{S3_NLP}imdb'
        # self.path= untar_data(IMDB, dest=path)

        self.path = Path(path) 
        print('path:', self.path)
        # download pre-trained language model if not already downloaded
        # URLs.download_wt103_model()
        self.path.mkdir(exist_ok=True)
        model_path = self.path/'models'
        model_path.mkdir(exist_ok=True)
        url = 'http://files.fast.ai/models/wt103_v1/'
        download_url(f'{url}lstm_wt103.pth', model_path/'lstm_wt103.pth')
        download_url(f'{url}itos_wt103.pkl', model_path/'itos_wt103.pkl')
        

        
        

        # print('cuda.is_available: ', torch.cuda.is_available())
        # import sys
        # sys.exit(1)



        # self.vocab = Vocab(IMDB_PATH/'models')
        self.random_gen = np.random.RandomState(324657436)

    def clear(self, remember_train_std_if_supported=False):
        '''
        :param remember_train_std_if_supported: ignored
        '''
        if os.path.exists(self.path/'tmp'):
            shutil.rmtree(self.path/'tmp')

    def to_data_frame(self, pos_inst, neg_inst, shuffle=False):

        X = pos_inst + neg_inst
        y = np.asarray([1] * len(pos_inst) + [0] * len(neg_inst))

        yX = list(zip(y, X))
        if shuffle:
            self.random_gen.shuffle(yX)

        return pd.DataFrame(yX)


    def set_train_data(self, pos_inst, neg_inst):
        '''
        :param pos_inst: list of strings
        :param neg_inst: list of strings
        :return:
        '''
        self.train_size = len(pos_inst) + len(neg_inst)
        train_df = self.to_data_frame(pos_inst, neg_inst, shuffle=True)
        train_df.to_csv(self.path/'train.csv', header=False, index=False, quoting=csv.QUOTE_NONNUMERIC)

    def set_valid_data(self, pos_inst, neg_inst):
        '''
        :param pos_inst: list of strings
        :param neg_inst: list of strings
        :return: labels of instances in the order they are kept here (corresponds with predict())
        '''
        self.valid_size = len(pos_inst) + len(neg_inst)
        valid_df = self.to_data_frame(pos_inst, neg_inst, shuffle=False)
        valid_df.to_csv(self.path/'valid.csv', header=False, index=False, quoting=csv.QUOTE_NONNUMERIC)
        return [1]*len(pos_inst)+[0]*len(neg_inst)



    def train_best(self):
        

        self.clear()

        # Following:
        # https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson3-imdb.ipynb

        # Average dev F1 score for episode of size 20 0.4988081516106912 stdev 0.2008833526477335 accuracy 0.546325167037862 stdev 0.04830498278899628
        # Average dev F1 score for episode of size 60 0.5806918220082198 stdev 0.13782926466261117 accuracy 0.6177431328878991 stdev 0.055230510152611996
        # Average dev F1 score for episode of size 200 0.7167359812579885 stdev 0.04195987855969531 accuracy 0.7324424647364516 stdev 0.028046644432594157
        # Average dev F1 score for episode of size 400 0.7867103107324568 stdev 0.029486784046659564 accuracy 0.7887156644394951 stdev 0.022195365297312083   
        bs=8
        self.data_in_lm_format = TextLMDataBunch.from_csv(self.path, bs=bs)
        
        # fine-tune the pre-trained language model using the train data
        tuned_lm = RNNLearner.language_model(self.data_in_lm_format, pretrained_fnames=['lstm_wt103', 'itos_wt103'], drop_mult=0.5)
        # tuned_lm.lr_find()
        # Recorder.plot(skip_end=15)
        # import sys
        # sys.exit(1)

        lr_multiply = 2

        tuned_lm.fit_one_cycle(1, lr_multiply*1e-2, moms=(0.8,0.7))
        
        tuned_lm.save('fit_head')
        tuned_lm.load('fit_head')

        tuned_lm.unfreeze()
        tuned_lm.fit_one_cycle(4, lr_multiply*1e-3, moms=(0.8,0.7))
        
        tuned_lm.save('fine_tuned')
        tuned_lm.load('fine_tuned')

        tuned_lm.save_encoder(self.path/'lm_fine_tuned')
    

        self.data_in_clas_format = TextClasDataBunch.from_csv(self.path, bs=bs, vocab=self.data_in_lm_format.train_ds.vocab)

        self.classifier = RNNLearner.classifier(self.data_in_clas_format, drop_mult=0.5)
        self.classifier.load_encoder(self.path/'lm_fine_tuned')
        self.classifier.freeze()
        # self.classifier.lr_find()
        self.classifier.fit_one_cycle(1, lr_multiply*2e-2, moms=(0.8,0.7))
        
        self.classifier.save('first')
        self.classifier.load('first')

        self.classifier.freeze_to(-2)
        self.classifier.fit_one_cycle(1, slice(lr_multiply*1e-2/(2.6**4),lr_multiply*1e-2), moms=(0.8,0.7))
        
        self.classifier.save('second')
        self.classifier.load('second')

        self.classifier.freeze_to(-3)
        self.classifier.fit_one_cycle(1, slice(lr_multiply*5e-3/(2.6**4),lr_multiply*5e-3), moms=(0.8,0.7))
        
        self.classifier.save('third')
        self.classifier.load('third')

        self.classifier.unfreeze()
        self.classifier.fit_one_cycle(15, slice(lr_multiply*1e-3/(2.6**4),lr_multiply*1e-3), moms=(0.8,0.7))

    def train3(self):

        self.clear()

        bs=48
        self.data_in_lm_format = TextLMDataBunch.from_csv(self.path, bs=bs)
        
        # fine-tune the pre-trained language model using the train data
        tuned_lm = RNNLearner.language_model(self.data_in_lm_format, pretrained_fnames=['lstm_wt103', 'itos_wt103'], drop_mult=0.4)
        # tuned_lm.lr_find()
        # Recorder.plot(skip_end=15)
        # import sys
        # sys.exit(1)

        # tuned_lm.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
        tuned_lm.fit_one_cycle(1, 1e-2)
        
        tuned_lm.save('fit_head')
        tuned_lm.load('fit_head')

        tuned_lm.unfreeze()
        # tuned_lm.fit_one_cycle(4, 1e-3, moms=(0.8,0.7))
        tuned_lm.fit_one_cycle(4, 1e-3)
        
        tuned_lm.save('fine_tuned')
        tuned_lm.load('fine_tuned')

        tuned_lm.save_encoder(self.path/'lm_fine_tuned')
    

        self.data_in_clas_format = TextClasDataBunch.from_csv(self.path, bs=bs, vocab=self.data_in_lm_format.train_ds.vocab)

        self.classifier = RNNLearner.classifier(self.data_in_clas_format, drop_mult=0.5)
        self.classifier.load_encoder(self.path/'lm_fine_tuned')
        self.classifier.freeze()
        # self.classifier.lr_find()
        # self.classifier.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
        self.classifier.fit_one_cycle(1, 2e-2)
        
        self.classifier.save('first')
        self.classifier.load('first')

        self.classifier.freeze_to(-2)
        # self.classifier.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
        self.classifier.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2))
        
        self.classifier.save('second')
        self.classifier.load('second')

        self.classifier.freeze_to(-3)
        # self.classifier.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
        self.classifier.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))
        
        self.classifier.save('third')
        self.classifier.load('third')

        self.classifier.unfreeze()
        # self.classifier.fit_one_cycle(10, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
        self.classifier.fit_one_cycle(10, slice(1e-3/(2.6**4),1e-3))



    def train2_1(self):
        self.train2(drop_mult_lm=0.5, drop_mult_clf=0.5, cycles_lm=4, cycles_clf=10, bs=8)

    def train2_2(self):
        self.train2(drop_mult_lm=0.5, drop_mult_clf=0.5, cycles_lm=2, cycles_clf=10, bs=8)


    def train2_treat(self):
        self.train2(drop_mult_lm=0.3, drop_mult_clf=0.5, cycles_lm=5, cycles_clf=5, bs=48)
    # def train2_treat(self):
        # self.train2(drop_mult_lm=0.3, drop_mult_clf=0.5, cycles_lm=10, cycles_clf=2, bs=48)


    def train2(self, drop_mult_lm, drop_mult_clf, cycles_lm, cycles_clf, bs):
        
        self.clear()

        # Following:
        # https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson3-imdb.ipynb

        # if self.train_size >= 100:
        #     batch_size = 70
        # else:
        #     batch_size = max(self.train_size//2, 16)
        # print('Using batch size', batch_size)
        # data_in_lm_format = TextLMDataBunch.from_csv(self.path, bs=batch_size)
        # data_in_clas_format = TextClasDataBunch.from_csv(self.path, vocab=data_in_lm_format.train_ds.vocab, bs=batch_size)
        
        self.data_in_lm_format = TextLMDataBunch.from_csv(self.path, bs=bs)
        
        # fine-tune the pre-trained language model using the train data
        tuned_lm = RNNLearner.language_model(self.data_in_lm_format, pretrained_fnames=['lstm_wt103', 'itos_wt103'], drop_mult=drop_mult_lm)
        # tuned_lm.lr_find()
        # Recorder.plot(skip_end=15)
        # import sys
        # sys.exit(1)

        tuned_lm.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
        
        tuned_lm.save('fit_head')
        tuned_lm.load('fit_head')

        tuned_lm.unfreeze()
        tuned_lm.fit_one_cycle(cycles_lm, 1e-3, moms=(0.8,0.7))
        
        tuned_lm.save('fine_tuned')
        tuned_lm.load('fine_tuned')

        tuned_lm.save_encoder(self.path/'lm_fine_tuned')
    

        self.data_in_clas_format = TextClasDataBunch.from_csv(self.path, bs=bs, vocab=self.data_in_lm_format.train_ds.vocab)

        self.classifier = RNNLearner.classifier(self.data_in_clas_format, drop_mult=drop_mult_clf)
        self.classifier.load_encoder(self.path/'lm_fine_tuned')
        self.classifier.freeze()
        # self.classifier.lr_find()
        self.classifier.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
        
        self.classifier.save('first')
        self.classifier.load('first')

        self.classifier.freeze_to(-2)
        self.classifier.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
        
        self.classifier.save('second')
        self.classifier.load('second')

        self.classifier.freeze_to(-3)
        self.classifier.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
        
        self.classifier.save('third')
        self.classifier.load('third')

        self.classifier.unfreeze()
        self.classifier.fit_one_cycle(cycles_clf, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))




    def train_baseline(self):

        self.clear()

        # Following:
        # https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson3-imdb.ipynb

        # if self.train_size >= 100:
        #     batch_size = 70
        # else:
        #     batch_size = max(self.train_size//2, 16)
        # print('Using batch size', batch_size)
        # data_in_lm_format = TextLMDataBunch.from_csv(self.path, bs=batch_size)
        # data_in_clas_format = TextClasDataBunch.from_csv(self.path, vocab=data_in_lm_format.train_ds.vocab, bs=batch_size)
        
        bs=48
        self.data_in_lm_format = TextLMDataBunch.from_csv(self.path, bs=bs)
        
        # fine-tune the pre-trained language model using the train data
        tuned_lm = RNNLearner.language_model(self.data_in_lm_format, pretrained_fnames=['lstm_wt103', 'itos_wt103'], drop_mult=0.3)
        # tuned_lm.lr_find()
        # Recorder.plot(skip_end=15)
        # import sys
        # sys.exit(1)

        tuned_lm.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
        # tuned_lm.fit_one_cycle(1, 1e-3, moms=(0.8,0.7))

        tuned_lm.save('fit_head')
        tuned_lm.load('fit_head')

        tuned_lm.unfreeze()
        tuned_lm.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))
        # tuned_lm.fit_one_cycle(10, 1e-4, moms=(0.8,0.7))

        tuned_lm.save('fine_tuned')
        tuned_lm.load('fine_tuned')

        tuned_lm.save_encoder(self.path/'lm_fine_tuned')
    

        self.data_in_clas_format = TextClasDataBunch.from_csv(self.path, bs=bs, vocab=self.data_in_lm_format.train_ds.vocab)

        self.classifier = RNNLearner.classifier(self.data_in_clas_format, drop_mult=0.5)
        self.classifier.load_encoder(self.path/'lm_fine_tuned')
        self.classifier.freeze()
        # self.classifier.lr_find()
        self.classifier.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
        # self.classifier.fit_one_cycle(1, 2e-3, moms=(0.8,0.7))

        self.classifier.save('first')
        self.classifier.load('first')

        self.classifier.freeze_to(-2)
        self.classifier.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
        # self.classifier.fit_one_cycle(1, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))

        self.classifier.save('second')
        self.classifier.load('second')

        self.classifier.freeze_to(-3)
        self.classifier.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
        # self.classifier.fit_one_cycle(1, slice(5e-4/(2.6**4),5e-4), moms=(0.8,0.7))

        self.classifier.save('third')
        self.classifier.load('third')

        self.classifier.unfreeze()
        self.classifier.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
        # self.classifier.fit_one_cycle(2, slice(1e-4/(2.6**4),1e-4), moms=(0.8,0.7))

    def train_fast_dummy(self):

        self.clear()      
        bs=48
        self.data_in_lm_format = TextLMDataBunch.from_csv(self.path, bs=bs)
        
        # fine-tune the pre-trained language model using the train data
        tuned_lm = RNNLearner.language_model(self.data_in_lm_format, pretrained_fnames=['lstm_wt103', 'itos_wt103'], drop_mult=0.3)
  
        tuned_lm.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
  
        tuned_lm.save_encoder(self.path/'lm_fine_tuned')
    

        self.data_in_clas_format = TextClasDataBunch.from_csv(self.path, bs=bs, vocab=self.data_in_lm_format.train_ds.vocab)

        self.classifier = RNNLearner.classifier(self.data_in_clas_format, drop_mult=0.5)
        self.classifier.load_encoder(self.path/'lm_fine_tuned')
        # self.classifier.freeze()
        # self.classifier.lr_find()
        self.classifier.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
        # self.classifier.fit_one_cycle(1, 2e-3, moms=(0.8,0.7))


    def train(self): 
        self.train2_treat()
        # self.train_fast_dummy()
       
    def predict(self, is_test=False):
        #classifier.get_preds returns [outputs, input_labels]
        print('computing predictions')
        preds = self.classifier.get_preds(is_test)
        preds[0] = torch.nn.functional.softmax(preds[0], dim=1) # turn raw scores to probabilities that add up to 1

        label_ind_preds = torch.argmax(preds[0], dim=1).numpy()
        label_preds = self.data_in_clas_format.classes[label_ind_preds]

        label1_ind = -1
        for i, label in enumerate(self.data_in_clas_format.classes):
            if label == 1: # POSITIVE LABEL
                label1_ind = i
        assert(label1_ind >= 0)

        label1_scores = preds[0][:,label1_ind]

        # label_scores, label_preds = torch.max(preds[0], dim=1)



        
        label1_scores = label1_scores.numpy()
        # label_preds = label_preds.numpy()

           # ds = data.train_ds
        # vocab_size, lbl = ds.vocab_size, ds.labels[0]



        #re-ordering predictions according to original order as loaded
        sampler = list(self.data_in_clas_format.test_dl.sampler if is_test else self.data_in_clas_format.valid_dl.sampler)
        ordered_label_preds = [None]*len(label_preds)
        ordered_label1_scores = [None]*len(label1_scores)
        for i in range(len(sampler)):
            ordered_label_preds[sampler[i]] = label_preds[i] 
            ordered_label1_scores[sampler[i]] = label1_scores[i] 
        return ordered_label_preds, ordered_label1_scores


