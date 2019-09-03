import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert import BertModel
# from pytorch_pretrained_bert.modeling import PreTrainedBertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from enum import Enum

class BertPoolType(Enum):
    
    AVG = 1
    MAX = 2
    RNN = 3
    


def token_logits_2_sent_weights(logits):
    ''' computes average probability that a token is a rationale word across a sentence '''
    scores_per_token = torch.nn.functional.softmax(logits, dim=2)[:,:,1]
    scores_per_sent = torch.mean(scores_per_token, dim=1, keepdim=False)
    scores_per_sent = scores_per_sent/torch.sum(scores_per_sent)
    return scores_per_sent


class BertForLongSequenceModel(BertPreTrainedModel):
    '''
    This model uses BERT to represent a set of sentences and then classifies them using LongSequenceClassifierComponent
    '''
    def __init__(self, bert_config, pool_type=BertPoolType.AVG, num_labels=2, reduced_hidden_size=None, upper_dropout=None, shallow_fine_tuning=True):
        '''
        :param bert_config
        :param pool_type: how to encode the entire instance based on its component sentences
        :param num_labels: number of labels to classify to
        :param reduced_hidden_size: seq classifier config param
        :param upper_dropout: seq classifier config param
        '''
        super(BertForLongSequenceModel, self).__init__(bert_config)
        self.shallow_fine_tuning = shallow_fine_tuning
        if upper_dropout == None: 
            upper_dropout = bert_config.hidden_dropout_prob
        self.seq_classifier = LongSequenceClassifierComponent(bert_config.hidden_size, upper_dropout, pool_type, num_labels, reduced_hidden_size)
        self.bert = BertModel(bert_config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, avg_weights=None):
        '''
        :param input_ids: token ids [sent_num x sent_max_length]
        :param token_type_ids: token type ids (only 'A' segment used) [sent_num x sent_max_length]
        :param attention_mask: which tokens are to be considered (i.e. not PAD tokens) [sent_num x sent_max_length]
        :param labels: the label of this entire instance [1]
        :return loss if labels are provided, otherwise logits
        '''
        token_level_reps, whole_sentence_reps = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sentence_reps = token_level_reps[:,0,:] if self.shallow_fine_tuning else whole_sentence_reps
        return self.seq_classifier(sentence_reps, token_type_ids, attention_mask, labels, avg_weights=avg_weights)
        

class BertForRationaleClassification(BertPreTrainedModel):
    '''
    This model uses BERT to classify tokens to rationale labels
    '''
    
    def __init__(self, bert_config, num_labels, upper_dropout=None):
        '''
        :param bert_config
        :param num_labels: number of labels to classify to
        :param upper_dropout: param for RationaleClassifierComponent
        '''
        super(BertForRationaleClassification, self).__init__(config)
        if upper_dropout == None: upper_dropout = config.hidden_dropout_prob
        self.rationale_classifier = RationaleClassifierComponent(num_labels, upper_dropout, config.hidden_size)
        self.bert = BertModel(bert_config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        '''
        :param input_ids: token ids [sent_num x sent_max_length]
        :param token_type_ids: token type ids (only 'A' segment used) [sent_num x sent_max_length]
        :param attention_mask: which tokens are to be considered (i.e. not PAD tokens) [sent_num x sent_max_length]
        :param labels: the label for each token [sent_num x sent_max_length]
        :return loss if labels are provided, otherwise logits
        '''
        token_level_reps, whole_sentence_reps = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)        
        return self.rationale_classifier(token_level_reps, attention_mask, labels)


class BertForTextClassificationWithRationalesModel(BertPreTrainedModel):
    '''
    This model learns to classify texts with the help of rationale word classification as an auxiliary task
    '''
    
    def __init__(self, bert_config, pool_type=BertPoolType.AVG, num_text_labels=2, 
    reduced_hidden_size=None, upper_dropout=None, use_rationales=False, is_multi_label_rationales=False, 
    learn_weights=False, shallow_fine_tuning=True):
        super(BertForTextClassificationWithRationalesModel, self).__init__(bert_config)
        '''
        :param bert_config
        :param pool_type: how to encode the entire instance based on its component sentences
        :param num_labels: number of labels to classify to
        :param reduced_hidden_size: seq classifier config param
        :param upper_dropout: seq classifier config param
        :param use_rationales: whether to use rationale classification or not
        :param is_multi_label_rationales: whether to label rationale words differently per each rationale (False in paper)
        :param learn_weights: whether to learn an attention model for sentence weights
        '''
        self.bert = BertModel(bert_config)
        self.shallow_fine_tuning = shallow_fine_tuning
        self.use_rationales = use_rationales
        
        if upper_dropout == None: 
            upper_dropout = bert_config.hidden_dropout_prob
        
        self.seq_classifier = LongSequenceClassifierComponent(bert_config.hidden_size, upper_dropout, pool_type, num_text_labels, 
                        reduced_hidden_size, learn_attn_weights=learn_weights)
        num_rationale_labels = num_text_labels + is_multi_label_rationales
        self.rationale_classifier = RationaleClassifierComponent(num_rationale_labels, upper_dropout, bert_config.hidden_size) if use_rationales else None
        
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, text_labels=None, rationale_labels=None, avg_weights=None):
        '''
        :param input_ids: token ids [sent_num x sent_max_length]
        :param token_type_ids: token type ids (only 'A' segment used) [sent_num x sent_max_length]
        :param attention_mask: which tokens are to be considered (i.e. not PAD tokens) [sent_num x sent_max_length]
        :param text_labels: the target label of the entire instance [1]
        :param rationale_labels: the label for each token [sent_num x sent_max_length]
        :param avg_weights: (optional) weights to be used when averaging the sentences representations
        :return loss if labels are provided, otherwise logits
        '''
        assert text_labels is None or rationale_labels is None, "labels input not valid"
        
        token_level_reps, whole_sentence_reps = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # using middle layers to infuse rationale supervision to lower layers of BERT
        # token_level_reps_list, whole_sentence_reps = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        # middle_layer = len(token_level_reps_list) // 2 - 1 # 6th layer (ind=5)
        # token_level_reps = token_level_reps_list[middle_layer]

        sentence_reps = token_level_reps[:,0,:] if self.shallow_fine_tuning else whole_sentence_reps

        if text_labels is None and rationale_labels is None: # predict text labels and rationale labels
            if self.use_rationales:
                rationale_logits = self.rationale_classifier(token_level_reps, attention_mask, None)
                avg_weights = token_logits_2_sent_weights(rationale_logits)
            else:
                rationale_logits = None
                avg_weights = None

            return self.seq_classifier(sentence_reps, token_type_ids, attention_mask, None, avg_weights=avg_weights), avg_weights, rationale_logits 
                
        elif text_labels is not None: # text classification train
            return self.seq_classifier(sentence_reps, token_type_ids, attention_mask, text_labels, avg_weights)
        else: # rationale classification train
            return self.rationale_classifier(token_level_reps, attention_mask, rationale_labels)
        

class LongSequenceClassifierComponent(nn.Module):
    '''
    Classifies a text instance represented by a set of sentences that are already encoded
    '''
    
    def __init__(self, input_size, dropout_prob, pool_type=BertPoolType.AVG, num_labels=2, reduced_hidden_size=None, learn_attn_weights=False):
        super(LongSequenceClassifierComponent, self).__init__()
        '''
        :param input_size: units num in the sentence encoding
        :dropout_prob: dropout to apply
        :pool_type: how to encode the entire instance based on its component sentences
        :param num_labels: number of labels to classify to
        :param reduced_hidden_size: used for BertPoolType.RNN
        :param learn_attn_weights: learn attention weights for weighted sentence averaging
        '''

        if learn_attn_weights:
            assert pool_type == BertPoolType.AVG
        self.pool_type = pool_type
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_prob)
        if pool_type == BertPoolType.RNN:
            rnn_hidden_size = reduced_hidden_size if reduced_hidden_size != None else input_size
            self.rnn = nn.LSTM(input_size, rnn_hidden_size, 1, dropout=dropout_prob)
            self.classifier = nn.Linear(rnn_hidden_size, num_labels)
        else:
            self.classifier = nn.Linear(input_size, num_labels)
        self.weight_model = nn.Linear(input_size, 1) if learn_attn_weights else None
        

    def forward(self, sequences_reps, token_type_ids=None, attention_mask=None, labels=None, avg_weights=None):
        '''
        :param sequences_reps: encoding of the sentences (sequences) [num_sents x encoding_size]
        :param token_type_ids: token type ids (only 'A' segment used) [sent_num x sent_max_length]
        :param attention_mask: which tokens are to be considered (i.e. not PAD tokens) [sent_num x sent_max_length]
        :param labels: the label of this entire instance [1]
        :param avg_weights: (optional) weights to be used when averaging the sentences representations
        :return loss if labels are provided, otherwise logits
        '''

        assert(not(self.pool_type != BertPoolType.AVG and avg_weights is not None))
        assert not(self.weight_model is not None and avg_weights is not None)

        sequences_reps = self.dropout(sequences_reps)

        if self.weight_model: # use the learned attention model to compute avg_weights
            avg_weights = self.weight_model(sequences_reps)
            avg_weights = torch.nn.functional.softmax(avg_weights, dim=0)
            sequences_reps = avg_weights * sequences_reps
            long_seq_rep = torch.sum(sequences_reps, dim=0, keepdim=True)
        elif self.pool_type == BertPoolType.AVG:
            if avg_weights is not None: # use input avg_weights
                sequences_reps = (avg_weights * sequences_reps.t()).t()
                long_seq_rep = torch.sum(sequences_reps, dim=0, keepdim=True)
            else: # use uniform weights
                long_seq_rep = torch.mean(sequences_reps, dim=0, keepdim=True)
        elif self.pool_type == BertPoolType.MAX:
            long_seq_rep = torch.max(sequences_reps, dim=0, keepdim=True)[0]
        elif self.pool_type == BertPoolType.RNN:
            hidden = None
            sequences_reps = torch.reshape(sequences_reps, (sequences_reps.size(0), 1, sequences_reps.size(1)))
            output, hidden = self.rnn(sequences_reps, hidden)
            long_seq_rep = output[output.size(0)-1] # the output of the last rnn step  
        else:
            raise Exception("Pool type not supported:", self.pool_type)

        logits = self.classifier(long_seq_rep)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class RationaleClassifierComponent(nn.Module):

    '''
    Classifies input tokens already encoded
    '''

    def __init__(self, num_labels, dropout_prob, input_size):
        '''
        :param num_labels: number of labels to classify to
        :dropout_prob: dropout to apply
        :param input_size: input encoding size
        '''
        super(RationaleClassifierComponent, self).__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(input_size, num_labels)
        
    def forward(self, token_reps, attention_mask=None, labels=None):
        '''
        :param token_reps: the encoding of each input token [sent_num x sent_max_length x encoding_size]
        :param attention_mask: which tokens are to be considered (i.e. not PAD tokens) [sent_num x sent_max_length]
        :param labels: the label for each token [sent_num x sent_max_length]
        :return loss if labels are provided, otherwise logits
        '''

        token_reps = self.dropout(token_reps)
        logits = self.classifier(token_reps)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits

#---------------------------------------------------
# The models below were not include in the paper since they are more complex, include more parameters 
# and did not outperform significantly the simpler models above on the dev set
#---------------------------------------------------

class BertForSequencesWeights(BertPreTrainedModel):
    '''
    This model uses its own independent BERT model to learn how to assign weight to a sentence (sequence)
    (This was not included in the paper)
    '''

    def __init__(self, config, upper_dropout=None):
        super(BertForSequencesWeights, self).__init__(config)
        if upper_dropout == None: upper_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(upper_dropout)
        self.weight_model = nn.Linear(config.hidden_size, 1)
        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        token_level_reps, whole_sentence_reps = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False) 
        sequences_reps = self.dropout(token_level_reps[:,0,:])
        avg_weights = self.weight_model(sequences_reps).t()
        avg_weights = torch.nn.functional.softmax(avg_weights, dim=1)
        return avg_weights


class TwoBertsForTextClassificationWithRationalesModel(nn.Module):
    '''
    This model uses two instance of the BERT model, each fine-tuned independently
    One BERT is used to classify texts to target classification labels
    Another BERT is used to classify tokens to rationales or as an attention weight model
    (This was not included in the paper)
    '''
    def __init__(self, pretrained_bert, pool_type=BertPoolType.AVG, num_text_labels=2, 
    reduced_hidden_size=None, upper_dropout=None, use_rationales=False, is_multi_label_rationales=False, learn_weights=False):

        assert (use_rationales and not learn_weights) or (not use_rationales and learn_weights)
        
        super(TwoBertsForTextClassificationWithRationalesModel, self).__init__()
        self.seq_classifier = BertForLongSequenceModel.from_pretrained(pretrained_bert, pool_type=pool_type, num_labels=num_text_labels, 
        reduced_hidden_size=reduced_hidden_size, upper_dropout=upper_dropout)

        if use_rationales:
            num_rationale_labels = num_text_labels + is_multi_label_rationales
            self.rationale_classifier = BertForRationaleClassification.from_pretrained(pretrained_bert, num_labels=num_rationale_labels, upper_dropout=upper_dropout)
            self.weight_model = None
        else: # learn weights
            self.weight_model = BertForSequencesWeights.from_pretrained(pretrained_bert, upper_dropout=upper_dropout)
            self.rationale_classifier = None

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, text_labels=None, rationale_labels=None, avg_weights=None):

        assert text_labels is None or rationale_labels is None, "labels input not valid"
        # if not(text_labels==None or rationale_labels==None):
            # raise Exception("labels input not valid") # not training both objectives in the same function call

        if text_labels is None and rationale_labels is None: # predict text labels (and rationale labels)
            assert avg_weights is None
            if self.rationale_classifier is not None:
                rationale_logits = self.rationale_classifier(input_ids, token_type_ids, attention_mask)
                rational_scores_per_sent = token_logits_2_sent_weights(rationale_logits)
                avg_weights = rational_scores_per_sent
            else:
                avg_weights = self.weight_model(input_ids, token_type_ids, attention_mask)

            return self.seq_classifier(input_ids, token_type_ids, attention_mask, labels=None, avg_weights=avg_weights)

        elif text_labels is not None: # text classification train

            # assert avg_weights is not None 
            # return self.seq_classifier.forward(whole_sentence_reps, token_type_ids, attention_mask, text_labels, avg_weights)
            if self.weight_model is not None:
                assert(avg_weights is None)
                avg_weights = self.weight_model(input_ids, token_type_ids, attention_mask)
            return self.seq_classifier(input_ids, token_type_ids, attention_mask, labels=text_labels, avg_weights=avg_weights)
        else: # rationale classification train
            return self.rationale_classifier(input_ids, token_type_ids, attention_mask, rationale_labels)
        


