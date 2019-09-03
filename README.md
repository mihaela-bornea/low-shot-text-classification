# low-shot-text-classification

Running the experiments:

RB-BOW-PROTO
python few_shot_experiment.py --model PROTOTYPE --rep W2V --norm True --bias 6 --min_word_count 0 --output experiment_w2v.txt --embeddings /Users/mabornea/data/embeddings/GoogleNews-vectors-negative300.bin.onewords.txt --data_dir data --dataset movie_reviews --ep_iter_count 30

BOW-PROTO
python few_shot_experiment.py --model PROTOTYPE --rep W2V --norm True --bias 0 --min_word_count 0 --output experiment_w2v.txt --embeddings /Users/mabornea/data/embeddings/GoogleNews-vectors-negative300.bin.onewords.txt --data_dir data --dataset movie_reviews --ep_iter_count 30

RB-BOW-SVM
python few_shot_experiment.py --model RB_SVM --rep W2V --norm True --bias 0 --min_word_count 0 --output experiment_w2v.txt --embeddings /Users/mabornea/data/embeddings/GoogleNews-vectors-negative300.bin.onewords.txt --data_dir data --dataset movie_reviews --ep_iter_count 30

SVM
python few_shot_experiment.py --model SVM --rep ONE_HOT --norm True --bias 0 --min_word_count 0 --output experiment_w2v.txt --embeddings /Users/mabornea/data/embeddings/GoogleNews-vectors-negative300.bin.onewords.txt --data_dir data --dataset movie_reviews --ep_iter_count 30 --idf True


RA-SVM
python few_shot_experiment.py --model SVM --rep ONE_HOT --norm True --bias 0 --min_word_count 0 --output experiment_w2v.txt --embeddings /Users/mabornea/data/embeddings/GoogleNews-vectors-negative300.bin.onewords.txt --data_dir data --dataset movie_reviews --ep_iter_count 30 --idf True --nrfd 0.1


RB_WAVG_BERT
python few_shot_experiment.py --model BERT --bert_pool_type AVG --one_bert --bert_rationale_weight 1.0 --bert_text_weight 1.0 --bert_detach_weights --lr 5e-6 --dropout 0.1 --epochs 10 --rep NA --store_dir store/ --output experiment_results.txt  --data_dir data/ --dataset movie_reviews --ep_iter_count 30 