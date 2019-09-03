'''
Wrapper for elmo embeddings
'''

import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from embedding.embeddings_cache import EmbeddingsCache

default_options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
default_weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

class ElmoEmbeddings:

    def __init__(self, options_file=default_options_file, weight_file=default_weight_file, normalize=True, gpu=-1, cache=None):
        '''
        :param options_file: elmo options file
        :param weight_file: elmo weight file
        :param normalize: whether to L2 normalize the embedding vectors to 1.0
        '''

        self.gpu = gpu
        with torch.cuda.device(self.gpu):
            self.elmo = Elmo(options_file, weight_file, 2, dropout=0)
            self.normalize = normalize
            self.cache = cache


    def is_context_sensitive(self):
        return True

    def is_seq_embedder(self):
        return False

    def size(self):
        return -1 # elmo is character-based, no limit on vocab size

    def units(self):
        return 1024

    def __contains__(self, w):
        return True

    def represent_text(self, text):

        with torch.cuda.device(self.gpu):

            if self.cache != None:
                text_key = '\t'.join(text)

            if self.cache != None and text_key in self.cache:
                    avg_embeddings = self.cache.text2rep(text_key)
            else:
                character_ids = batch_to_ids([text]) # doing batch size 1 for now
                elmo_layers = torch.stack(self.elmo(character_ids)['elmo_representations'], dim=0)
                avg_embeddings = torch.mean(elmo_layers,0) # even average of elmo layers
                avg_embeddings = avg_embeddings.squeeze(dim=0) # remove the batch size 1 dimension
                if self.cache != None:
                    self.cache.add_entry(text_key, avg_embeddings)

            if self.normalize:
                norm = torch.norm(avg_embeddings, p=2, dim=1, keepdim=True)
                avg_embeddings = avg_embeddings.div(norm.expand_as(avg_embeddings))

            return avg_embeddings.detach().cpu().numpy()


# # use batch_to_ids to convert sentences to character ids
# sentences = [['First', 'sentence', '.'], ['Another', '.']]
# character_ids = batch_to_ids(sentences)
# embeddings = elmo(character_ids)
# print(embeddings['elmo_representations'][0].shape)
#
# # embeddings['elmo_representations'] is length two list of tensors.
# # Each element contains one layer of ELMo representations with shape
# # (2, 3, 1024).
# #   2    - the batch size
# #   3    - the sequence length of the batch
# #   1024 - the length of each ELMo vectors