from data_utils.data_manager import DataManager
from prototype.prototype import Prototype
from embedding.embeddings import Embeddings
from embedding.embeddings_service import EmbeddingsService
#from embedding.elmo_embeddings import ElmoEmbeddings
from embedding.bert_embeddings import BertEmbeddings
from embedding.embeddings_cache import EmbeddingsCache
import numpy as np
import torch
from spacy.lang.en.stop_words import STOP_WORDS
from classifiers.binary_classifier import BinaryClassifier, CLASSIFIER_TYPE
#from classifiers.ulmfit_classifier import UlmfitClassifier
from classifiers.bert_classifier import BertClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from embedding.embeddings_service import TextSpan
import collections
from enum import Enum
import datetime
import argparse
import random
from classifiers.bert_custom_models import BertPoolType
from classifiers.bert_classifier import BertTrainConfig

class RATIONALE_REP(Enum):
    SINGLETON = 1 #This has a single representation for all rationales
    PER_CLASS = 2 #This has a representation for positives and a representation for negatives
    PER_INSTANCE = 3 #This experiment has a representation for each rationale

class MODEL_TYPE(Enum):
    PROTOTYPE = 1
    PA = 2
    LR = 3
    SVM = 4
    ULMFIT = 5
    BERT = 6
    RBSVM = 7


class WORD_REP(Enum):
    NA = 0
    W2V = 1
    ONE_HOT = 2
    ELMO = 3
    BERT = 4

random_gen = np.random.RandomState(3333)

def rationale_representation(train_set, emb_service, aggregate = True, freq_words = None ):
    rationales = []
    for t_i in train_set:
        #print("inst " + str(t_i))
        t_i_text = t_i["text"]
        t_i_sentences = t_i["sents"]
        t_i_spans = t_i["spans"]
        t_i_rationale_words = []
        #print(t_i_spans)
        for s in t_i_sentences:
            s_start = s[0]
            s_end = s[1]
            s_text = t_i_text[s_start:s_end]
            for t_i_span in t_i_spans:
                if s_start <= t_i_span[0] and s_end>=t_i_span[1] :
                    token_start = len(t_i_text[s_start:t_i_span[0]].split())
                    token_end = token_start + len(t_i_text[t_i_span[0] : t_i_span[1]].split())
                    t_i_rationale_words.append(TextSpan(s_text.split(), begin=token_start, end=token_end))
        if len(t_i_rationale_words) > 0 :
            rationales.append(emb_service.represent_spans(t_i_rationale_words,use_words = freq_words))
    if aggregate :
        rationales_vec = emb_service.centroid(rationales)
        return rationales_vec
    else :
        return rationales

def instance_split(inst):
    inst_text = inst["text"]
    inst_sentences = inst["sents"]
    if len(inst_sentences) == 0:
        print('NOTICE: no sentences in inst', inst['rid'])
    inst_spans = []
    for s in inst_sentences :
        s_text = inst_text[s[0]:s[1]]
        inst_spans.append(TextSpan(s_text.split()))
    return inst_spans

def instance_split_with_rationales(inst,down_weight, up_weight=1):
    inst_text = inst["text"]
    inst_sentences = inst["sents"]
    inst_spans = inst["spans"] #rationales
    #print(t_i_spans)
    inst_text_spans = []
    for s in inst_sentences:
        s_start = s[0]
        s_end = s[1]
        s_text = inst_text[s_start:s_end]
        s_weights = [down_weight] * len(s_text.split())
        for i_span in inst_spans:
            if s_start <= i_span[0] and s_end>=i_span[1] :
                token_start = len(inst_text[s_start:i_span[0]].split())
                token_end = token_start + len(inst_text[i_span[0] : i_span[1]].split())
                for i in range(token_start, token_end):
                    s_weights[i]=up_weight
        inst_text_spans.append(TextSpan(s_text.split(), weights=s_weights))
    return inst_text_spans

def text_representation(train_set, emb_service, use_words ):
    text_vecs = []
    for t_i in train_set:
        t_i_text = t_i["text"]
        t_i_sents = t_i["sents"]
        t_i_spans = []
        for s in t_i_sents :
            s_text = t_i_text[s[0]:s[1]]
            t_i_spans.append(TextSpan(s_text.split()))
        text_vecs.append(emb_service.represent_spans(t_i_spans, use_words = use_words))
    return emb_service.centroid(text_vecs)

def sent_lens(data_set):

    under30 = 0
    under46 = 0
    under62 = 0
    under94 = 0
    longer = 0
    for t_i in data_set:
        t_i_sents = t_i["sents"]
        t_i_text = t_i["text"]
        for s in t_i_sents:
            length = len(t_i_text[s[0]:s[1]].split())
            if length <=30:
                under30 += 1
            elif length <= 46:
                under46 += 1
            elif length <= 62:
                under62 += 1
            elif length <= 94:
                under94 += 1
            else:
                longer += 1
                
    return under30, under46, under62, under94, longer

def sents_per_instance(data_set):

    result = []
    for t_i in data_set:
        result.append(len(t_i["sents"]))
    print(sorted(result))



def  print_instances (instances):
    for inst in instances :
        print ("RID " + str(inst["rid"]))
        text = inst["text"]
        print  ("Text: " + text)
        print ("Label: " + str(inst["label"]))
        for span in inst["spans"] :
            print ("Rationale: " + text[span[0]:span[1]])
        print ("\n")

def predict_random(instances):
    instance_predictions = [ 1 if x >= 0.5 else 0 for x in random_gen.random_sample(len(instances))]
    return instance_predictions


def evaluate(predictions, labels):
    _,_,f1,_=precision_recall_fscore_support(labels,predictions)
    acc = accuracy_score(labels,predictions)
    return acc,f1[1] #this is the positive class

def get_word_doc_counts(instances):
    word_counts = collections.Counter()
    for inst in instances:
        words_set = set(inst["text"].split())
        word_counts.update(words_set)
    return word_counts


def get_frequent_words(instances, min_freq):
    word_counts = get_word_doc_counts(instances)
    # word_counts = collections.Counter()
    # for inst in instances:
    #     words_set = set(inst["text"].split())
    #     word_counts.update(inst["text"].split())
    freq_words = set()
    for word, count in word_counts.most_common():
        if count < min_freq:
            break
        else:
            freq_words.add(word)
    return freq_words


def inst2text_with_capitals(inst):   
    text = inst["text"]
    sents = inst["sents"]
    capitalized_sents = [text[s[0]:s[1]].capitalize() for s in sents]
    return ' '.join(capitalized_sents)

def inst2sents(inst):   
    text = inst["text"]
    sents = inst["sents"]
    sents = [text[s[0]:s[1]] for s in sents]
    return sents

def read_word_counts(word_counts_file):

    print('Reading word counts from:'+word_counts_file)

    counts = {}
    with open(word_counts_file, 'r') as f:
        for line in f:
            segs = line.strip().split('\t')
            word = segs[0]
            count = float(segs[1])
            counts[word]=count

    return counts

def get_idfs(word_counts, doc_num):

    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
    # idf(d, t) = log [ (1 + n) / (1 + df(d, t)) ] + 1

    idfs = {}
    # total_counts = sum(word_counts.values())

    for word in word_counts.keys():
        idfs[word] = np.log((1+float(doc_num)) / (1+float(word_counts[word])))+1
    min_idf = min(idfs.values())
    print('Min idf: %.2f. Fixing to 1.0' % min_idf)
    for word in idfs.keys():
        idfs[word] = idfs[word]-min_idf+1

    print('Finished computing idf values for %d words.' % len(idfs))

    return idfs

def dump_predictions(pred_file, dataset_name, data, dev_predictions, dev_pos_predictions_scores):

    if dev_pos_predictions_scores == None:
        dev_pos_predictions_scores= [0]*len(data)
    with open(args.pred_file, 'a') as f:
        f.write('DATASET: '+dataset_name+'\n')
        for inst, score in zip(data, dev_pos_predictions_scores):
            f.write('%s\t%f\n' % (inst['rid'], score))

def train_and_evaluate(cls, data_manager, train_sample, pos_train_sample, neg_train_sample, emb_service, freq_words, debug_text):

    if model == MODEL_TYPE.PROTOTYPE:

        rationale_vecs = []
        if rationale_rep ==  RATIONALE_REP.SINGLETON :
            rationale_vecs.append(rationale_representation(train_sample, emb_service, freq_words=freq_words))
        if rationale_rep == RATIONALE_REP.PER_CLASS :
            rationale_vecs.append(rationale_representation(neg_train_sample, emb_service, freq_words=freq_words))
            rationale_vecs.append(rationale_representation(pos_train_sample, emb_service, freq_words=freq_words))
        if rationale_rep == RATIONALE_REP.PER_INSTANCE :
            rationale_vecs = rationale_representation(train_sample, emb_service, aggregate=False, freq_words =freq_words)

        #Adjust the bias(rationale representation) by removing the general text representation
        text_vec = text_representation(train_sample, emb_service, use_words=freq_words)
        rationale_vecs = [emb_service.add_vecs(rationale_vec, -text_vec) for rationale_vec in rationale_vecs]

        prototype = Prototype(emb_service)
        pos_vecs = [emb_service.represent_spans(instance_split(inst), use_words=freq_words, bias_vecs = rationale_vecs) for inst in pos_train_sample]
        pos_prototype = prototype.build_prototype(pos_vecs)
        neg_vecs = [emb_service.represent_spans(instance_split(inst), use_words=freq_words, bias_vecs = rationale_vecs) for inst in neg_train_sample]
        neg_prototype = prototype.build_prototype(neg_vecs)

        #The id of each class prototype is the label of the class
        #neg_prototype has label 0
        #pos_prototype has label 1
        class_prototypes = [neg_prototype,pos_prototype]

        # if args.text != None: # for debug
        if debug_text != None:
            text_vec, weights = emb_service.represent_span(use_words=freq_words, text=args.text.split(), begin=0, end=-1, bias_vecs=rationale_vecs, return_weights=True, weights=None)
            weights = [w**bias_strength for w in weights]
            # if norm:
            #     norm2 = np.sqrt(sum([w**2 for w in weights]))
            #     weights = [w/norm2 for w in weights]
            
            text_pred = prototype.apply_prototype([text_vec], class_prototypes)[0]
            # print('Text: %s' % args.text)
            # print('Weights: %s' % ' '.join([str(w) for w in weights]))
            print('Prediction: %d' % text_pred)
            print('\n'.join(['%s\t%.8f' % (word, weight) for (word, weight) in zip(debug_text.split(), weights)]))
            
            import sys
            sys.exit(0)
        else:
            dev_vecs = [emb_service.represent_spans(instance_split(inst), use_words=freq_words, bias_vecs = rationale_vecs) for inst in data_manager.get_data()]
            dev_predictions, dev_pos_predictions_scores = prototype.apply_prototype(dev_vecs, class_prototypes)
            dev_labels = np.array([inst["label"] for inst in data_manager.get_data()])
            acc,f1 = evaluate(dev_predictions,dev_labels)
                
    
    elif model == MODEL_TYPE.RBSVM :
    
        rationale_vecs = []
        rationale_vecs.append(rationale_representation(neg_train_sample, emb_service, freq_words=freq_words))
        rationale_vecs.append(rationale_representation(pos_train_sample, emb_service, freq_words=freq_words))
     
        #Adjust the bias(rationale representation) by removing the general text representation
        text_vec = text_representation(train_sample, emb_service, use_words=freq_words)
        rationale_vecs = [emb_service.add_vecs(rationale_vec, -text_vec) for rationale_vec in rationale_vecs]

        pos_vecs = [emb_service.represent_spans(instance_split(inst), use_words=freq_words, bias_vecs = rationale_vecs) for inst in pos_train_sample]

        neg_vecs = [emb_service.represent_spans(instance_split(inst), use_words=freq_words, bias_vecs = rationale_vecs) for inst in neg_train_sample]
   
        cls.clear(remember_train_std_if_supported=True)
        cls.add_positive_instances(pos_vecs)
        cls.add_negative_instances(neg_vecs)
        cls.train()          
        
        dev_vecs = [emb_service.represent_spans(instance_split(inst), use_words=freq_words, bias_vecs = rationale_vecs) for inst in data_manager.get_data()]         
        dev_predictions = cls.predict(dev_vecs)
        dev_pos_predictions_scores = None # TODO: extract this info if possible
        dev_labels = np.array([inst["label"] for inst in data_manager.get_data()])
        acc,f1 = evaluate(dev_predictions,dev_labels)

    #model != MODEL_TYPE.PROTOTYPE :
    elif model == MODEL_TYPE.BERT:

        label_list = [0,1]
        label2id = {label : i for i, label in enumerate(label_list)}

        train_examples = [inst2sents(inst) for inst in train_sample]
        train_labels = [inst["label"] for inst in train_sample]
        if args.bert_rationale_weight > 0:
            # for every word in input sentences the label would be '1' if the word was marked as rationale and '0' otherwise
            train_token_label_ids = [[text_span.weights for text_span in 
            # multi-label rationales
            # instance_split_with_rationales(inst,down_weight=0, up_weight=label2id[inst["label"]]+1)] for inst in train_sample]
            instance_split_with_rationales(inst,down_weight=0, up_weight=1)] for inst in train_sample]
            # dev_token_label_ids = [[text_span.weights for text_span in instance_split_with_rationales(inst,down_weight=0)] for inst in data_manager.get_data()]
        else:
            train_token_label_ids = None
            # dev_token_label_ids = None

        train_config = BertTrainConfig(num_train_epochs=args.epochs, learning_rate=args.lr, upper_dropout=args.dropout, 
                        max_seq_length=args.bert_max_seq_len)
        bert_pool_type = BertPoolType[args.bert_pool_type]
        cls.train(bert_pool_type, train_examples, train_labels, label_list=label_list, label2id=label2id, train_token_label_ids=train_token_label_ids, 
            train_config=train_config, rationale_weight=args.bert_rationale_weight, text_weight=args.bert_text_weight, 
            two_berts=args.two_berts, detach_weights=args.bert_detach_weights, learn_weights=args.bert_learn_weights, 
            bert_independent_rationales=args.bert_independent_rationales, shallow_fine_tuning=not args.bert_deep_fine_tuning)
        
        dev_examples = [inst2sents(inst) for inst in data_manager.get_data()]
        dev_labels = np.array([inst["label"] for inst in data_manager.get_data()])
        dev_predictions, dev_pos_predictions_scores = cls.predict(dev_examples)
        acc,f1 = evaluate(dev_predictions, dev_labels)
            
    elif model == MODEL_TYPE.ULMFIT:

        train_pos_texts = [inst2text_with_capitals(inst) for inst in pos_train_sample]
        train_neg_texts = [inst2text_with_capitals(inst) for inst in neg_train_sample]
        dev_pos_texts = [inst2text_with_capitals(inst) for inst in data_manager.get_data() if inst["label"]==1]
        dev_neg_texts = [inst2text_with_capitals(inst) for inst in data_manager.get_data() if inst["label"]==0]

        cls.set_train_data(train_pos_texts, train_neg_texts)
        dev_labels = cls.set_valid_data(dev_pos_texts, dev_neg_texts)
        # print('NOT DUMPING TRAIN/DEV DATA!!!!!')
        cls.train()
        dev_predictions, dev_pos_predictions_scores = cls.predict()
        acc,f1 = evaluate(dev_predictions, dev_labels)

    else:

        cls.clear(remember_train_std_if_supported=True)

        if args.nrfd > 0 :
            pos_vecs = [emb_service.represent_spans(instance_split_with_rationales(inst, args.nrfd), use_words=freq_words) for inst in pos_train_sample]
            neg_vecs = [emb_service.represent_spans(instance_split_with_rationales(inst, args.nrfd), use_words=freq_words) for inst in neg_train_sample]
        else :
            pos_vecs = [emb_service.represent_spans(instance_split(inst), use_words=freq_words) for inst in pos_train_sample]
            neg_vecs = [emb_service.represent_spans(instance_split(inst), use_words=freq_words) for inst in neg_train_sample]
        dev_vecs = [emb_service.represent_spans(instance_split(inst), use_words=freq_words) for inst in data_manager.get_data()]
        dev_labels = np.array([inst["label"] for inst in data_manager.get_data()])
        
        cls.add_positive_instances(pos_vecs)
        cls.add_negative_instances(neg_vecs)
        cls.train()                   
        dev_predictions = cls.predict(dev_vecs)
        dev_pos_predictions_scores = None # TODO: extract this info if possible
        acc,f1 = evaluate(dev_predictions, dev_labels)    
        
    return acc, f1, dev_predictions, dev_pos_predictions_scores   

if __name__ == '__main__':

    print('\n'+str(datetime.datetime.now()))

    #np.random.seed(98634975)
    #torch.manual_seed(744875692)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="Random seed", type=int, default=98634975)
    parser.add_argument("--norm", help="Normalize the embeddings", type=bool, default=True)
    parser.add_argument("--sow", help="Set-of-words representation (only zero/one counts of words in a given text)", type=bool, default=False)
    parser.add_argument("--model", help="The type of model", type=str, choices=[i.name for i in MODEL_TYPE])
    parser.add_argument("--bert_pretrained_name", help="The pretrained BERT model to load", type=str, default='bert-base-uncased')
    parser.add_argument("--bert_pool_type", help="The way sentences are pooled together in the BERT model", type=str, choices=[i.name for i in BertPoolType])
    parser.add_argument("--bert_rationale_weight", help="Weight of rationales in bert training (0 means none)", type=float, default=0.0)
    parser.add_argument("--bert_text_weight", help="Weight of instance labels loss in bert training (0 means none)", type=float, default=1.0)
    parser.add_argument("--two_berts", dest='two_berts', help="Use a separate bert for label and rationale predictions", action='store_true')
    parser.add_argument("--one_bert", dest='two_berts', help="Use a single bert for both label and rationale predictions", action='store_false')
    parser.add_argument("--bert_detach_weights", dest='bert_detach_weights', help="Bert token/sent weights are not learned from instance labels", action='store_true')
    parser.add_argument("--bert_attach_weights", dest='bert_detach_weights', help="Bert token/sent weights are learned (also) from instance labels", action='store_false')
    parser.add_argument("--bert_learn_weights", help="Learn sentence weights from instance labels (like attention)", action='store_true')
    parser.add_argument("--bert_independent_rationales", help="Rationales will not be used for weighted averaging in classification", action='store_true')
    parser.add_argument("--bert_max_seq_len", help="Texts longer than that (in terms of word-piece count) will be truncated", type=int, default=48)
    parser.add_argument("--bert_deep_fine_tuning", help="Fine tune a linear layer on top of a pooling layer for sentence classification", action='store_true')

    parser.add_argument("--rep", help="The type of the representation", type=str, choices=[i.name for i in WORD_REP])
    parser.add_argument("--bias", help="The strength of the rationale", type=int)
    parser.add_argument("--min_word_count", help="The min count for a word to be considered", type=int, default=0)
    parser.add_argument("--output", help="The output file for the experiments", type=str)
    parser.add_argument("--embeddings", help="The path to the embeddings file", type=str, default=None)
    parser.add_argument("--idf", help="Apply inverse-doc-frequence weights", type=bool, default=False)
    parser.add_argument("--idf_file", help="The path to the inverse-doc-frequence file. If None, then the train set is used.", type=str, default=None)
    parser.add_argument("--elmo_cache", help="The path to the elmo cache file", type=str, default=None)
    parser.add_argument("--ep_iter_count", help="The number of experiment iterations per episode", type=int, default=30)
    parser.add_argument("--ep_sizes", help="List of the training sizes to be used", type=str, default="2 6 10 20 60 200 400")
    parser.add_argument("--nrfd",help="The weight discount for rationale classifiers", type=float,default=0.0)
    parser.add_argument("--data_dir", help="The path to input files", type=str, default="data")
    parser.add_argument("--dataset", help="Name of dataset to run the experiments on", type=str, default=None)
    parser.add_argument("--store_dir", help="The path to store files (e.g. ulmfit data, bert models)", type=str, default=None)
    parser.add_argument("--text", help="Input text to classify for debug (instead of running on dev/test sets)", type=str, default=None)
    parser.add_argument("--pred_file", help="Output file to dump classifier predictions to", type=str, default=None)
    parser.add_argument("--epochs", help="Number of training epochs (currently used only for bert)", type=int, default=3)
    parser.add_argument("--lr",help="Learning rate for training (currently used only for bert)", type=float,default=5e-6)
    parser.add_argument("--dropout",help="Probability to drop", type=float,default=0.1)
    parser.add_argument("--no_sent_split", help="Ignore sentence split in dataset (treats instance as one continguous text", action='store_true')
    
    

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    num_iter_per_episode = args.ep_iter_count

    print('two_berts', args.two_berts)

    print('num_iter_per_episode: %d' % num_iter_per_episode)

    rationale_rep = RATIONALE_REP.PER_CLASS

    model = MODEL_TYPE[args.model]

    #model = MODEL_TYPE.PROTOTYPE
    word_rep = WORD_REP[args.rep] if args.rep is not None else None
    #word_rep = WORD_REP.W2V

    norm = args.norm
    #norm = 1

    sow = args.sow

    bias_strength = args.bias
    #bias_strength = 6

    word_feature_min_freq = args.min_word_count
    #word_feature_min_freq = 0

    output = open(args.output, "a")
    predictions_output = open(args.pred_file, 'w') if args.pred_file != None else None
    #output = open("experiments.txt", "w")

    nrfd_string = "" if args.nrfd == 0 else "_nrfd_"+str(args.nrfd)
    idf_string = "" if args.idf == False else  "_idf"

    data_dir = args.data_dir

    conf = ("Experiment Setup:\nmodel = " + str(model)+ "\nword representation  = "+str(word_rep)+" norm = "+str(norm)
            +"\nbias_strength = "+str(bias_strength)+"\nword_feature_min_freq = " + str(word_feature_min_freq))

    id = (str(model)+"_"+str(word_rep)+"_n_"+str(int(norm))+"_b_"+str(bias_strength)+"_wfc_"+str(word_feature_min_freq)+nrfd_string+idf_string)

    print (id)

    output.write("\n\n\n")
    output.write(conf)
    output.write("\n")
    output.write(id)
    output.write("\n")


    cache = None
    stats_counts = False
    # print('Stats counting is ON!')

    if model == MODEL_TYPE.ULMFIT or model == MODEL_TYPE.BERT:
        emb_service = None
        print("NOTICE: Embedding service is not used with model type", model)

    else:

        idfs = None    

        if args.idf == True and args.idf_file != None:
            # For now, this is not word doc counts, but simply word counts
            counts = read_word_counts(args.idf_file)
            idfs = get_idfs(counts, sum(counts.values()))
        
        if word_rep == WORD_REP.ONE_HOT or word_rep == WORD_REP.W2V:
            # embedding_path = '/data7/DrWatson/embeddings/google/GoogleNews-vectors-negative300.bin.onewords.txt'
            embeddings = Embeddings(args.embeddings, normalize=norm, one_hot=(word_rep==WORD_REP.ONE_HOT), stats_count=stats_counts)  # embeddings file is used also for one-hot just to read the vocabulary
        elif word_rep == WORD_REP.ELMO:
            # cache = EmbeddingsCache('/users/Oren.Melamud/data/fewshot/movie_review_elmo_cache.bin')
            cache = EmbeddingsCache(args.elmo_cache)
            embeddings = ElmoEmbeddings(normalize=norm, cache=cache)
        elif word_rep == WORD_REP.BERT:
            embeddings = BertEmbeddings(cache_dir=args.store_dir+'/bert/', stats_count=stats_counts)
            
        emb_service = EmbeddingsService(embeddings, normalize=norm, bias_strength=bias_strength, stopwords=STOP_WORDS,
                                        sow_representation=args.sow, idfs=idfs)

        print("Finished initializing embedding service")


    episodes = [int(s) for s in args.ep_sizes.strip().split()]
    print('Episode sizes:', episodes)

    
    cls = None

    if model == MODEL_TYPE.LR :
        cls = BinaryClassifier(CLASSIFIER_TYPE.LR)
    elif model == MODEL_TYPE.PA :
        cls = BinaryClassifier(CLASSIFIER_TYPE.PA)
    elif model == MODEL_TYPE.RBSVM :
        cls = BinaryClassifier(CLASSIFIER_TYPE.SVM)
    elif model == MODEL_TYPE.SVM :
        cls = BinaryClassifier(CLASSIFIER_TYPE.SVM)
    elif model == MODEL_TYPE.ULMFIT :
        cls = UlmfitClassifier(args.ulmfit_dir)
    elif model == MODEL_TYPE.BERT:
        cls = BertClassifier(pretrained_bert=args.bert_pretrained_name, num_labels=2)

    data_manager = DataManager(data_dir+"/"+args.dataset+".train.json",
                data_dir+"/"+args.dataset+".dev.json",
                data_dir+"/"+args.dataset+".test.json",
                0, dev = False, sent_split = not args.no_sent_split)
    print('Using dataset: ', args.dataset)
    #experiment = {}

    print('\n'+str(datetime.datetime.now()))

    for ep_sz in episodes :
        f1_scores_per_ep = []
        acc_scores_per_ep = []
        #get num_iter_per_episode samples
        
        #train_samples_per_episode = []
        
        output.write(str(ep_sz)+" examples for training")
        output.write("\n")

        if cls != None :
            cls.clear()

        for n_iter in range(num_iter_per_episode):

            pos_train_sample = []
            neg_train_sample = []
            train_sample = data_manager.get_train_sample(ep_sz)
            train_sample_ids = [ts["rid"] for ts in train_sample]
            print("TRAIN-SET SIZE: "+str(ep_sz)+"\tITER: " + str(n_iter))
            output.write("Train_sample: " + id +" " +str(ep_sz) + " "+ str(train_sample_ids))
            output.write("\n")

            # for ONE_HOT we use freq_words always to restrict the vocab only to the words in the train-set (all other words are meaningless in one-hot)
            freq_words = get_frequent_words(train_sample, word_feature_min_freq) if (word_rep==WORD_REP.ONE_HOT or word_feature_min_freq > 1) else None
            
            if args.idf == True and args.idf_file == None:
                word_counts = get_word_doc_counts(train_sample)
                idfs = get_idfs(word_counts, len(train_sample)) 
                emb_service.set_idfs(idfs)

            for ts in train_sample :
                if ts["label"] == 0 :
                    neg_train_sample.append(ts)
                else :
                    pos_train_sample.append(ts)

            acc = None
            f1 = None

            if len(pos_train_sample) > 0 and len(neg_train_sample) > 0 :

                acc,f1, dev_predictions, dev_pos_predictions_scores = train_and_evaluate(cls, data_manager, train_sample, pos_train_sample, neg_train_sample, emb_service, freq_words, args.text)                    

            else :

                print("Ep size " + str(ep_sz) + " sample number " + str(n_iter) + " random class assignment")
                dev_predictions = predict_random(data_manager.get_data())
                dev_pos_predictions_scores = None
                dev_labels = np.array([inst["label"] for inst in data_manager.get_data()])
                acc,f1 = evaluate(dev_predictions, dev_labels)


            f1_scores_per_ep.append(f1)
            acc_scores_per_ep.append(acc)

            #output.write('iter acc ' + str(acc) + ' f1 ' + str(f1) + '\n')
            print('iter acc ' + str(acc) + ' f1 ' + str(f1) + '\n')

            if args.pred_file != None:
                dump_predictions(predictions_output, args.dataset, data_manager.get_data(), dev_predictions, dev_pos_predictions_scores)

        output.write("F1: " + id + " " +str(ep_sz) + " " + str(f1_scores_per_ep))
        output.write("\n")
        output.write("Accuracy: " + id + " " +str(ep_sz) + " " + str(acc_scores_per_ep))
        output.write("\n")
        
        #experiment[str(ep_sz)] = train_samples_per_episode

        output.write("Avg F1 " + id + " " +str(ep_sz) + " " + str(np.average(np.array(f1_scores_per_ep))) + " stdev " +str(np.std(f1_scores_per_ep))+"\n")
        output.write ("Avg accuracy " + id + " " +str(ep_sz) + " " + str(np.average(np.array(acc_scores_per_ep))) +" stdev " +str(np.std(acc_scores_per_ep)) + "\n\n" )

        print ("Average dev F1 score for episode of size " + str(ep_sz) + " " + str(np.average(np.array(f1_scores_per_ep))) + " stdev " + str(np.std(f1_scores_per_ep)) + " accuracy " + str(np.average(np.array(acc_scores_per_ep))) +" stdev " + str(np.std(acc_scores_per_ep)) )

        
        if stats_counts:
            print('total_toks=%d, unk_toks=%d, unk_ratio=%.3f' % (emb_service.embeddings.total_toks, emb_service.embeddings.unks, emb_service.embeddings.get_unk_ratio()))

    if cache != None:
        cache.close()
    print('\n'+str(datetime.datetime.now())+'\n')
    output.close()
    if predictions_output != None:
        predictions_output.close()
