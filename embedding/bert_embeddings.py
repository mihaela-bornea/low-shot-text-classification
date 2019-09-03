'''
Wrapper for bert embeddings
'''

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

class BertEmbeddings:

    def __init__(self, model_name='bert-base-uncased', cache_dir=None, max_seq_length=64, max_batch_size=64, stats_count=False):
        '''
        :param normalize: whether to L2 normalize the embedding vectors to 1.0
        '''
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print('BertEmbeddings DEVICE: ', self.device)
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = BertModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.to(self.device)
        self.model.eval()

        #debug stats
        self.stats_count = stats_count
        if self.stats_count:
            self.unks = 0
            self.total_toks = 0

    def get_unk_ratio(self):
        return float(self.unks)/self.total_toks

    def is_context_sensitive(self):
        return False

    def is_seq_embedder(self):
        '''
        This embedder embed the entire text sequence into a single vector (not vector per word)
        '''
        return True

    def size(self):
        return -1 

    def units(self):
        return -1

    def __contains__(self, w):
        return True

    def tokenize_text(self, text):
        
        # Tokenized input
        tokenized_text = self.tokenizer.tokenize(' '.join(text))
        if len(tokenized_text) > self.max_seq_length-2:
            tokenized_text = tokenized_text[:self.max_seq_length-2]


        if self.stats_count:
            self.unks += tokenized_text.count('[UNK]')
            self.total_toks += len(tokenized_text)

        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        mask_ids = [1]*len(indexed_tokens)
        indexed_tokens.extend([0]*((self.max_seq_length-2)-len(indexed_tokens)))
        mask_ids.extend([0]*((self.max_seq_length-2)-len(mask_ids)))

        segments_ids = [0] * len(indexed_tokens)

        return indexed_tokens, segments_ids, mask_ids



    def represent_text_batch(self, text_batch): 

        represented_num = 0
        encoded_instances = []
        while represented_num < len(text_batch):
            n = min(self.max_batch_size, len(text_batch)-represented_num)  
            encoded_n = self.represent_small_text_batch(text_batch[represented_num:represented_num+n])
            encoded_instances.append(encoded_n) 
            represented_num += n

        if len(encoded_instances) > 1:
            # print('Large batch size:', len(text_batch))
            return np.concatenate(encoded_instances, axis=0)
        else:
            return encoded_instances[0]

            
    def represent_small_text_batch(self, text_batch):

        indexed_tokens_batch,  segments_ids_batch, mask_ids_batch = zip(*[self.tokenize_text(text) for text in text_batch])
        tokens_tensor = torch.tensor(indexed_tokens_batch, device=self.device)
        segments_tensor = torch.tensor(segments_ids_batch, device=self.device)
        masks_tensor = torch.tensor(mask_ids_batch, device=self.device)
        encoded_words, encoded_text = self.model(tokens_tensor, segments_tensor, attention_mask=masks_tensor, output_all_encoded_layers=False) 
        return encoded_text.detach().cpu().numpy()

    # def represent_text(self, text):

    #     with torch.cuda.device(self.gpu):

    #         # Tokenized input
    #         tokenized_text = self.tokenizer.tokenize(' '.join(text))

    #         # Convert token to vocabulary indices
    #         indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
    #         segments_ids = [0] * len(indexed_tokens)

    #         # Convert inputs to PyTorch tensors
    #         tokens_tensor = torch.tensor([indexed_tokens])
    #         segments_tensors = torch.tensor([segments_ids])

    #         # Predict hidden states features for each layer
    #         encoded_words, encoded_text = self.model(tokens_tensor, segments_tensors, output_all_encoded_layers=False)
            
    #         return encoded_text.detach().numpy()

if __name__ == '__main__':

    bert = BertEmbeddings()
    embeddings = bert.represent_text('This is a test yes')
    print(embeddings.shape)


    

