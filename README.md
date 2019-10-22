# low-shot-text-classification
This repository contains the code used in the paper:

**Combining Unsupervised Pre-training and Annotator Rationales to Improve Low-shot Text Classification**
Oren Melamud, Mihaela Bornea and Ken Barker Conference on Empirical Methods in Natural Language Processing (2019)

## Requirements

python 3.7  
pythorch 1.0  
pytorch-pretrained-bert 0.6.2  
spacy 2.0  

## Dataset Description
The data directory contains the train, dev and text files for the moview reviews(IMDB)  and aviation reports (ASRS) datasets.
For convenience we preprocessed the data and saved it in json format.

The original dataset can be fond at:
http://www.cs.jhu.edu/~ozaidan/rationales/

We used the pretrained embeddings from: 
https://code.google.com/archive/p/ word2vec/

## Running the Experiments:

### **RB_BOW_PROTO**

python few_shot_experiment.py --model PROTOTYPE --rep W2V --norm True --bias 6 --min_word_count 0 --output experiment_w2v.txt --embeddings <path_to_embeddings_file> --data_dir data --dataset <movie_reviews or aviation> --ep_iter_count 30

### **BOW_PROTO**

python few_shot_experiment.py --model PROTOTYPE --rep W2V --norm True --bias 0 --min_word_count 0 --output experiment_w2v.txt --embeddings <path_to_embeddings_file> --data_dir data --dataset <movie_reviews or aviation> --ep_iter_count 30

### **RB_BOW_SVM**

python few_shot_experiment.py --model RB_SVM --rep W2V --norm True --bias 0 --min_word_count 0 --output experiment_w2v.txt --embeddings <path_to_embeddings_file> --data_dir data --dataset <movie_reviews or aviation> --ep_iter_count 30

### **SVM**
 python few_shot_experiment.py --model SVM --rep ONE_HOT --norm True --bias 0 --min_word_count 0 --output experiment_w2v.txt --embeddings <path_to_embeddings_file> --data_dir data --dataset <movie_reviews or aviation> --ep_iter_count 30 --idf True


### **RA_SVM**

python few_shot_experiment.py --model SVM --rep ONE_HOT --norm True --bias 0 --min_word_count 0 --output experiment_w2v.txt --embeddings <path_to_embeddings_file> --data_dir data --dataset <movie_reviews or aviation> --ep_iter_count 30 --idf True --nrfd 0.1


### **RB_WAVG_BERT**

python few_shot_experiment.py --model BERT --bert_pool_type AVG --one_bert --bert_rationale_weight 1.0 --bert_text_weight 1.0 --bert_detach_weights --lr 5e-6 --dropout 0.1 --epochs 10 --rep NA --store_dir store/ --output experiment_results.txt  --data_dir data/ --dataset <movie_reviews or aviation> --ep_iter_count 30 

### **WAVG_BERT**

python few_shot_experiment.py  --model BERT --bert_pool_type AVG --one_bert --bert_rationale_weight 1.0 --bert_text_weight 1.0 --bert_detach_weights --bert_independent_rationales --lr 5e-6 --dropout 0.1 --epochs 10 --rep NA --store_dir store/ --output experiment_results.txt  --data_dir data/ --dataset <movie_reviews or aviation> --ep_iter_count 30
 