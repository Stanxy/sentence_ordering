# data processer
import json
from tqdm import tqdm
import random
from traceback import print_exc
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import itertools
from utils import warmup_linear, bundle_part_to_batch


class Bundle(object):
    """The structure to contain all data for training. 
    
    A flexible class. The properties are defined in FIELDS and dynamically added by capturing variables with the same names at runtime.
    """
    pass

class EncodingError(Exception):
    # The error used in the convert func
    pass

class LengthError(Exception):
    # The error used in the convert func
    pass



TRAIN_TEST_FIELDS = ['input_ids', 'masked_ids', 'token_type_ids', 'sep_positions', 'shuffled_index', \
                        'max_sample_len','ground_truth', 'passage_length', 'pairs_num', 'pairs_list', '_id']
PRED_FIELDS = ['input_ids', 'masked_ids', 'token_type_ids', 'sep_positions', 'max_sample_len', \
                        'shuffled_index', 'passage_length', 'pairs_num', 'pairs_list', '_id']

def fetch_sentence_pairs(tokenizer, pairs, shuffled_index, passage, p_key):
    ''' Fetch and process the input sentences and process them into the appropriate form for model1

        Args: 
            pairs (list of tuples) : the list of sentence pair combinations.
            shuffled_index (list): the list of shuffled index. A base index linked wit the sentence in passage
            passage (list of strings) : the list of sentences

        Returns:
            input_ids (list of list) : a list of list of shape [pairs_num, sequence_length]
            with the word token indices in the vocabulary
            masked_ids (list of list) : an optional list of list of shape [pairs_num, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
            token_type_ids (list of list) : an optional list of list of shape [pairs_num, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
            sep_positions (list of tuples) : a list of tuples of shape [pairs_num, 2] with the number denotes the position
            of the [SEP] tokens, this is for the last attention layers 
            max_sample_len (int): the max length of each pairs of sample
    '''
    input_ids = []
    masked_ids = []
    token_type_ids = []
    sep_positions = []

    max_sample_len = 0 # save the max length of ids inorder to do padding

    for sent_id1, sent_id2 in pairs:
        sep_position = (0,0)

        sent1 = passage[shuffled_index[sent_id1]]
        sent2 = passage[shuffled_index[sent_id2]]
        tokenized_sent1, tokenized_sent2 = tokenizer.tokenize(sent1), tokenizer.tokenize(sent2)

        concat_sents = ['[CLS]'] + tokenized_sent1 + ['[SEP]'] 
        sep_position[0] = len(concat_sents) - 1
        token_type_id = [0] * len(concat_sents)
        concat_sents = tokenized_sent2 + ['[SEP]']
        input_id = tokenizer.convert_tokens_to_ids(concat_sents)
        if len(concat_Sents) > 512: # 
            concat_Sents = concat_Sents[:512] # 
            print('SEQ TOO LONG! passage : {} pairs : {} {}'.format(p_key, \
                shuffled_index[sent_id1], shuffled_index[sent_id1]))

        sep_position[1] = len(concat_sents) - 1
        token_type_id = [1] * len(tokenized_sent2 + ['[SEP]'])
        masked_id = [1] * len(token_type_id)

        max_sample_len = len(masked_id) if max_sample_len < len(masked_id) else max_sample_len

        input_ids.append(input_id)
        masked_ids.append(masked_id)
        token_type_ids.append(token_type_id)
        sep_positions.append(sep_position)
    
    return input_ids, token_type_ids, masked_ids, sep_positions, max_sample_len

def pairs_generator(lenth):
    '''Generate the combinations of sentence pairs

        Args:
            lenth (int): the length of the sentence
        
        Returns:
            combs (list) : all combination of the index pairs in the passage
            num_combs (int) : the total number of all combs
    '''
    indices = list(range(lenth))
    combs_one_side = list(itertools.combinations(indices, 2))
    combs_other_side = [(x2, x1) for x1,x2 in combs_one_side]
    combs = ombs_oneside + combs_other_side
    return combs, len(combs)

def convert_passage_to_sample_bundle(tokenizer, data: 'json refined', mode, kw, p_key):
    '''Make training samples.
        
        Convert document-format dict samples(various fields plus a list or long sting as the passage) to bundles.
        
        Args:
            tokenizer (BertTokenizer): BERT Tokenizer to transform sentences to a list of word pieces.
            data (Json): metadata of passage in mogno document format. 
            mode (str): mode of the passage format, coule be a list (processed) or a long string (unprocessed).
            kw (str) : the key word map to the passage in each data dictionary. Defaults to 'abstract'
            p_key (str) : the key word to search for specific passage. Default to 'title'
        
        Raises:
            EncodingError: Invalid char type. 
            LengthError: Passage too short.

        Returns:
            Bundle: A bundle containing the fields for each sample (including gold and shaffled sample).
    '''
    if mode == 'list': # Then we need to clean data
        try:
            passage = data[kw].replace('\n',' ') # clean the '\n' char
            passage = ' '.join(word_tokenize(passage)) # clean the long spaces
            passage = sent_tokenize(passage)
        except:
            print ("Error when processing passage {}.\n".format(passage['id_'])) # print out which specific passage is dead
            print_exc()
    else:
        passage = data[kw]

    if len(passage) == 1:
        raise LengthError("The passage length is too short.")
    
    passage_length = len(passage) 
    base_index = list(range(passage_length)) 
    shuffled_index = base_index 
    random.shuffle(shuffled_index) 
    ground_truth = list(np.argsort(shuffled_index)) 
    pairs_list, pairs_num = pairs_generator(passage_length) 

    input_ids, token_type_ids, masked_ids, sep_positions, max_sample_len = \
        fetch_sentence_pairs(tokenizer, pairs, shuffled_index, passage, p_key)
    
    _id = data[p_key]
    ret = Bundle()
    for field in TRAIN_TEST_FIELDS: 
        setattr(ret, field, eval(field)) 
    return ret

def load_superbatch(data_file, super_batch_size): # we load the file by its superbatch into the RAM, instead of the whole file
    with open(data_file,'r') as f:
        count_line = 1 # initialize line
        lines = "" # initialize lines we want to load
        line_num = int(os.popen("wc -l " + data_file).read().split()[0]) # use wc -l to count the total lines
        for line in tqdm(f):

            if count_line % super_batch_size == 0 or count_line == line_num: # dump when a super is done
                lines += line.strip() # push the last line into the lines stack
                if lines.endswith(',\n'):
                    lines = lines[:-2]
                lines = '''[''' + lines + ''']''' # add 
                super_json_obj = json.loads(lines)
                yield super_json_obj, line_num # yield a list
                lines = "" # reinitialize lines

            lines += line
            count_line += 1

def homebrew_data_loader(bundles, batch_size = 8):
    '''To create a iterater from all samples 

    Inputs:
        bundles (list) : List of bundles.
        batch_size (int, optional): Defaults to 8. 

    Outputs:
        (int, Generator): number of batches and a generator to generate batches.
    '''

    bundles
    if hasattr(bundles[0], 'ground_truth'): # check if we are training or predict
        valid_field = 8
    else:
        valid_field = 7

    # here we only shuffle passage
    n = len(bundles)
    # Find how many samples are there in the list
    # random shuffle
    np.random.permutation(bundles)

    all_bundle = Bundle() # all_bundle is a bundle object
    for field in FIELDS[:valid_field]: # for the first 7 fields
        t = [] # 
        setattr(all_bundle, field, t) # t is an empty list. This is to set a bundle where its first 7 fields are all empty lists. 
        # seems like a kind of initialization
        for bundle in bundles: # iterate all bundle in bundles
            t.extend(getattr(bundle, field)) # 
        # In this step the fields in bundles are fill into an empty list 
    
    num_batch = (n - 1) // batch_size + 1
    def gen():
        for batch_num in range(num_batch):
            l, r = batch_num * batch_size, min((batch_num + 1) * batch_size, n) 
            # l and r is the start positon and end position.
            
            yield bundle_part_to_batch(all_bundle, l, r)
    return num_batch, gen()