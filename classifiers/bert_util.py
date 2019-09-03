import logging
import torch

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, token_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.token_label_ids = token_label_ids


def convert_instance_to_tensor(texts, label_id, word_label_ids, max_seq_length, tokenizer, device, max_texts=64):
    """
    :param texts: list of texts for the input instance (example)
    :param label_id: the single label id for the entire input instance 
    :param word_label_ids: if not None then contains a label for each word in every text
    :param max_seq_length: longer texts will be truncated
    :param tokenizer: used to tokenize words in the text to word-pieces
    :param device: gpu/cpu device to be used
    :param max_texts: instances with more texts than this will be truncated
    :return tensor representations of the inputs and labels
    """
    
    features = []
    token_label_ids = None
    for (ex_index, text) in enumerate(texts):

        if max_texts != None and ex_index >= max_texts:
            break

        token_label_ids = [0] if word_label_ids != None else None
        tokens = ["[CLS]"]
        
        if token_label_ids == None:
            tokens += tokenizer.tokenize(text)
        else:
            for word, word_label_id in zip(text.strip().split(), word_label_ids[ex_index]):
                pieces = tokenizer.tokenize(word)
                tokens += pieces
                token_label_ids += [word_label_id]*len(pieces)
        
        if len(tokens) > max_seq_length - 1:
            tokens = tokens[:(max_seq_length - 1)]
            if token_label_ids != None: token_label_ids = token_label_ids[:(max_seq_length - 1)]

        tokens += ["[SEP]"]
        if token_label_ids != None: token_label_ids += [0]
        
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        if token_label_ids != None: token_label_ids += padding
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert token_label_ids == None or len(token_label_ids) == max_seq_length

        if ex_index < 0:
            logger.info("*** Text ***")
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            if label_id != None:
                logger.info("label id = %d)" % (label_id))
            if token_label_ids != None:
                logger.info("token label id: %s" % " ".join([str(x) for x in token_label_ids]))

        features.append(
                InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id,
                            token_label_ids=token_label_ids))

    tensor_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long, device=device)
    tensor_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long, device=device)
    tensor_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long, device=device)
    tensor_label_id = torch.tensor([label_id], dtype=torch.long, device=device) if label_id != None else None
    tensor_token_label_ids = torch.tensor([f.token_label_ids for f in features], dtype=torch.long, device=device) if token_label_ids != None else None
    
    return tensor_input_ids, tensor_input_mask, tensor_segment_ids, tensor_label_id, tensor_token_label_ids
        
    
def str_token_preds(tokenizer, tensor_input_ids, token_scores):
    '''
    convert input_ids of tokens with their scores to a printable string
    '''
    text = ''
    for i in range(tensor_input_ids.size(0)):
        for j in range(tensor_input_ids.size(1)):
            # print('tok ids', tensor_input_ids[i][j].item())
            tokens = tokenizer.convert_ids_to_tokens([tensor_input_ids[i][j].item()])
            # print('tokens: ', tokens)
            token = tokens[0]
            # print('token:', tokens)
            if token == '[PAD]':
                break
            # tokenizer.convert_ids_to_tokens([tensor_input_ids[i][j].item()])[0]
            text += '%.4f:%s\t' % (token_scores[i][j], token)
        text += '\n'

    return text

def print_instance_scores(tokenizer, token_input_ids, token_scores, sentence_scores, instance_id, instance_score):

    '''
    convert input_ids of tokens with their scores to a printable string
    '''
    text = ''
    text += "INSTANCE ID: " + str(instance_id) + " SCORE: " + str(instance_score) + '\n'
    for i in range(token_input_ids.size(0)):
        text += '%.4f:%s\t' % (sentence_scores[i], "[AVERAGE]")
        for j in range(token_input_ids.size(1)):
            # print('tok ids', tensor_input_ids[i][j].item())
            tokens = tokenizer.convert_ids_to_tokens([token_input_ids[i][j].item()])
            # print('tokens: ', tokens)
            token = tokens[0]
            # print('token:', tokens)
            if token == '[PAD]':
                break
            # tokenizer.convert_ids_to_tokens([tensor_input_ids[i][j].item()])[0]
            text += '%.4f:%s\t' % (token_scores[i][j], token)
        text += '\n'

    return text



