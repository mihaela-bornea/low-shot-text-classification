# low-shot-text-classification
This repository contains the code used in the paper:

**Combining Unsupervised Pre-training and Annotator Rationales to Improve Low-shot Text Classification**
Oren Melamud, Mihaela Bornea and Ken Barker Conference on Empirical Methods in Natural Language Processing (2019)

## Requirements

* python 3.7  
* pythorch 1.0  
* pytorch-pretrained-bert 0.6.2  
* spacy 2.0  

## Dataset Description
The data directory contains the train, dev and text files for the moview reviews(IMDB)  and aviation reports (ASRS) datasets.
For convenience we preprocessed the data and saved it in json format.

We used the pretrained embeddings from: 
https://code.google.com/archive/p/word2vec/

The code requires the conversion of word vectors from the bin format to the txt format.
Bin to txt conversion tool: https://github.com/marekrei/convertvec

Convert from the txt to numpy format. 
``` 
python embedding/text2numpy.py <path_to_embeddings_file> <path_to_embeddings_file> 
```

We recommend that all embedding files are stored in the same folder. 
The scripts below use the path to the .txt embedding file.

```
ls embeddings
  GoogleNews-vectors-negative300.bin.txt		
  GoogleNews-vectors-negative300.bin.txt.vocab
  GoogleNews-vectors-negative300.bin.txt.npy
```

## Running the Experiments:

### **RB_BOW_PROTO**

```
python few_shot_experiment.py --model PROTOTYPE --rep W2V --norm True --bias 6 --min_word_count 0 --output <output_file> --embeddings <path_to_embeddings_file> --data_dir data --dataset <movie_reviews or aviation> --ep_iter_count 30
```

### **BOW_PROTO**

```
python few_shot_experiment.py --model PROTOTYPE --rep W2V --norm True --bias 0 --min_word_count 0 --output <output_file> --embeddings <path_to_embeddings_file> --data_dir data --dataset <movie_reviews or aviation> --ep_iter_count 30
```

### **RB_BOW_SVM**

```
python few_shot_experiment.py --model RB_SVM --rep W2V --norm True --bias 6 --min_word_count 0 --output <output_file> --embeddings <path_to_embeddings_file> --data_dir data --dataset <movie_reviews or aviation> --ep_iter_count 30
```

### **SVM**
 ```
 python few_shot_experiment.py --model SVM --rep ONE_HOT --norm True --bias 0 --min_word_count 0 --output <output_file> --embeddings <path_to_embeddings_file> --data_dir data --dataset <movie_reviews or aviation> --ep_iter_count 30 --idf True
```

### **RA_SVM**

```
python few_shot_experiment.py --model SVM --rep ONE_HOT --norm True --bias 0 --min_word_count 0 --output <output_file> --embeddings <path_to_embeddings_file> --data_dir data --dataset <movie_reviews or aviation> --ep_iter_count 30 --idf True --nrfd 0.1
```

### **RB_WAVG_BERT**

```
python few_shot_experiment.py --model BERT --bert_pool_type AVG --one_bert --bert_rationale_weight 1.0 --bert_text_weight 1.0 --bert_detach_weights --lr 5e-6 --dropout 0.1 --epochs 10 --rep NA --store_dir store/ --output <output_file>  --data_dir data/ --dataset <movie_reviews or aviation> --ep_iter_count 30 
```

### **WAVG_BERT**
```
python few_shot_experiment.py  --model BERT --bert_pool_type AVG --one_bert --bert_rationale_weight 1.0 --bert_text_weight 1.0 --bert_detach_weights --bert_independent_rationales --lr 5e-6 --dropout 0.1 --epochs 10 --rep NA --store_dir store/ --output <output_file>  --data_dir data/ --dataset <movie_reviews or aviation> --ep_iter_count 30
 ```
