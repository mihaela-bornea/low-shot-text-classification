import torch
# import apex
import random
import math

from pytorch_pretrained_bert.tokenization import BertTokenizer
from classifiers.bert_custom_models import BertForLongSequenceModel
from classifiers.bert_custom_models import BertForRationaleClassification
from classifiers.bert_custom_models import BertForTextClassificationWithRationalesModel
from classifiers.bert_custom_models import TwoBertsForTextClassificationWithRationalesModel
from classifiers.bert_custom_models import token_logits_2_sent_weights
# from pytorch_pretrained_bert.modeling import BertForTokenClassification

from pytorch_pretrained_bert.optimization import BertAdam

from classifiers.bert_util import convert_instance_to_tensor 
from classifiers.bert_util import str_token_preds
from classifiers.bert_util import print_instance_scores

from tqdm import tqdm, trange

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class BertTrainConfig:
    '''
    Config used to train (fine-tune) BERT-based models
    '''
    def __init__(self, num_train_epochs=3, learning_rate=5e-6, upper_dropout=0.1, max_seq_length=48):

        self.num_train_epochs = num_train_epochs
        # self.learning_rate = 5e-3
        self.learning_rate = learning_rate
        # self.learning_rate = (5e-5)/32
        self.upper_dropout = upper_dropout

        self.fp16 = False # not used
        self.loss_scale = 0
        self.warmup_proportion = 0.1
        self.gradient_accumulation_steps = 1 # not used
        self.max_seq_length = max_seq_length
        self.reduced_hidden_size = None
        

class BertClassifier:
    '''
    This class is used to train and apply various BERT-based classification models
    '''

    def __init__(self, pretrained_bert, num_labels):
        '''
        :param pretrained_bert: the name of the pretrained BERT to use
        :param num_labels: number of target text labels to classify to
        '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert, do_lower_case=True)
        self.pretrained_bert = pretrained_bert
        self.num_labels = num_labels
        
    def clear(self, remember_train_std_if_supported=False):
        '''
        :param remember_train_std_if_supported: ignored
        '''
        self.model = None
        
    @classmethod
    def warmup_linear(cls, x, warmup=0.002):
        if x < warmup:
            return x/warmup
        return 1.0 - x    
        

    @classmethod
    def get_optimizer(cls, model, num_train_steps, train_config):

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = num_train_steps
       
        optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=train_config.learning_rate,
                            warmup=train_config.warmup_proportion,
                            t_total=t_total)
        return optimizer


    
    def train(self, pool_type, train_examples, train_labels, label_list, label2id, train_token_label_ids=None, 
            train_config=None, rationale_weight=1.0, text_weight=1.0, two_berts=True, detach_weights=True, 
            learn_weights=False, bert_independent_rationales=False, shallow_fine_tuning=True):
        '''
        Trains a classifier
        @TODO groups these params somehow
        :param pool_type: how to encode the entire instance based on its component sentences
        :param train_examples: a list of train instances, each instance is a list of sentences (text sequences)
        :param train_labels: a list of labels for every train instance [num_instance]
        :param label_list: a list of all possible labels 
        :param label2id: maps labels to numerical indices
        :param train_token_label_ids: lists of token labels [num_instances x num_sent x num_token]
        :param train_config: the train config to use
        :param rationale_weight: the weight of the rationale classification task in the joint loss function
        :param text_weight: the weight of the text classification task in the joint loss function
        :param two_berts: whether to use two independent BERT models in the joint architecture (not included in paper)
        :param detach_weights: whether to block backprop from text loss from going to rationale classifier in the joint architecture
        :param learn_weights: whether to learn attention weights for sentence averaging
        :param bert_independent_rationales: if True then rationale training is only used to train the shared underlying BERT model 
                                            not for estimating sentence weights)
        '''


        if train_config == None:
            train_config = BertTrainConfig()
        self.train_config = train_config

        # TWO BERTS MODEL - This was not reported in the paper becaues these models are more complex, 
        # require more parameters and did not perform better on the dev sets.
        if two_berts:
            self.model = TwoBertsForTextClassificationWithRationalesModel(self.pretrained_bert, pool_type=pool_type, num_text_labels=2, 
                reduced_hidden_size=None, upper_dropout=None, use_rationales=(rationale_weight>0), 
                is_multi_label_rationales=False, learn_weights=learn_weights, shallow_fine_tuning=shallow_fine_tuning)
 
        # JOINT MODEL (ONE BERT)
        else:
            self.model = BertForTextClassificationWithRationalesModel.from_pretrained(self.pretrained_bert, \
                pool_type = pool_type, num_text_labels = self.num_labels, \
                reduced_hidden_size=train_config.reduced_hidden_size, upper_dropout=train_config.upper_dropout, \
                use_rationales=(rationale_weight>0), is_multi_label_rationales=False, learn_weights=learn_weights, shallow_fine_tuning=shallow_fine_tuning)


        self.model.to(self.device)   

        self.label_list = label_list
        self.label2id = label2id

        if train_examples == None: # just loading the pre-trained model
            return
        
        train_batch_size = 1 # one text at a time (will be converted into a batch of sentences)
        num_train_steps = int(len(train_examples) / train_batch_size * train_config.num_train_epochs)
        num_opt_steps = num_train_steps
        self.optimizer = BertClassifier.get_optimizer(self.model, num_opt_steps, train_config)

        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        # logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        self.model.train()
        for epoch in trange(int(train_config.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            tr_text_loss = 0
            tr_rationale_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # random.shuffle(train_examples)
            for step, instance in enumerate(train_examples):

                # modify learning rate with special warm up BERT uses
                lr_this_step = train_config.learning_rate * BertClassifier.warmup_linear(global_step/num_train_steps, train_config.warmup_proportion)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_this_step

                # only used if we're training with rationales
                token_label_ids = None if train_token_label_ids == None else train_token_label_ids[step]

                tensor_input_ids, tensor_input_mask, tensor_segment_ids, tensor_label_id, tensor_token_label_ids = \
                    convert_instance_to_tensor(instance, self.label2id[train_labels[step]], token_label_ids, train_config.max_seq_length, self.tokenizer, self.device)

                rationale_loss = 0
                rational_scores_per_sent = None
                if tensor_token_label_ids is not None: # rationale auxiliary loss
                    rationale_loss, rationale_logits = self.model(tensor_input_ids, tensor_segment_ids, tensor_input_mask, \
                        text_labels=None, rationale_labels=tensor_token_label_ids) 
                    if not bert_independent_rationales:
                        rational_scores_per_sent = token_logits_2_sent_weights(rationale_logits)

                    rationale_loss *=  rationale_weight  

                    tr_rationale_loss += rationale_loss.item()
                    tr_loss += rationale_loss.item()

                    
                if text_weight > 0:

                    if rational_scores_per_sent is not None and detach_weights:
                        rational_scores_per_sent = rational_scores_per_sent.detach()

                    text_loss = self.model(tensor_input_ids, tensor_segment_ids, tensor_input_mask, text_labels=tensor_label_id, 
                        rationale_labels=None, avg_weights=rational_scores_per_sent)
                    
                    text_loss *= text_weight
                    tr_text_loss += text_loss.item()
                    tr_loss += text_loss.item()
                    
  
                loss = rationale_loss+text_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                global_step += 1
                nb_tr_examples += tensor_input_ids.size(0)
                nb_tr_steps += 1
    
            mean_loss = tr_loss/nb_tr_steps
            mean_text_loss = tr_text_loss/nb_tr_steps
            mean_rationale_loss = tr_rationale_loss/nb_tr_steps
            logger.info("End of epoch %d. Mean loss: total %.4f text %.4f rationale %.4f", \
                epoch, mean_loss, mean_text_loss, mean_rationale_loss)
            

    def predict(self, test_examples):

        logger.info("***** Running predict *****")
        logger.info("  Num examples = %d", len(test_examples))

        self.model.eval()

        test_label_preds = []
        test_label1_scores = []

        for i, instance in enumerate(test_examples):

            tensor_input_ids, tensor_input_mask, tensor_segment_ids, _tensor_label_id, _tensor_token_label_ids= \
                convert_instance_to_tensor(instance, None, None, self.train_config.max_seq_length, self.tokenizer, self.device)             

            with torch.no_grad():
                logits, avg_weights, rationale_logits  = self.model(tensor_input_ids, tensor_segment_ids, tensor_input_mask)

            scores = torch.nn.functional.softmax(logits, dim=1)
            label_ind_preds = torch.argmax(scores, dim=1).detach().cpu().numpy()
            label_preds = [self.label_list[label_ind] for label_ind in label_ind_preds]

            label1_ind = self.label2id[1]
            label1_scores = scores[:,label1_ind]

            label1_scores = label1_scores.detach().cpu().numpy()

            # Use the code commented out below to print out internal sentence and token weights computed by the model
            # TODO - refactor this

            # if i < 10:
            #     # token_preds = torch.argmax(logits, dim=2).detach().cpu().numpy()
            #     rationale_scores = torch.nn.functional.softmax(rationale_logits, dim=2)
            #     rationale_positive_scores = rationale_scores[:,:,1]
            #     # logger.info("INSTANCE %d" % i)
            #     # logger.info(str_token_preds(self.tokenizer, tensor_input_ids.data, label1_scores))
            #     logger.info(print_instance_scores(self.tokenizer, tensor_input_ids.data, rationale_positive_scores, avg_weights, i, label1_scores))
            
            test_label_preds.append(label_preds)
            test_label1_scores.append(label1_scores)

        return test_label_preds, test_label1_scores

            
