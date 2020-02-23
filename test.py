# This file is the main process of the whole pipeline
import json
import os
from tqdm import tqdm, trange
import traceback
import numpy as np
# waiting to be finished
from data import convert_passage_to_samples_bundle, load_superbatch, homebrew_data_loader
from model import HierarchicalSentenceEncoder, GlobalGraphPropagation, BackwardFowardAttentiveDeoder
from model import calculate_loss, dev_test
from utils import warmup_linear, rouge_w, acc, kendall_tau, pmr

from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
import torch
from torch import nn
from torch.optim import Adam, Adadelta

def test(testing_data_file, super_batch_size, tokenizer, mode, kw, p_key, \
        model1, device, model2, model3):
    '''Train three models
    
    Train models through bundles
    
    Args:
        training_data_file (list) : training data json file, raw json file used to load data
        super_batch_size (int) : how many samples will be loaded into memory at once
        tokenizer : SentencePiece tokenizer used to obtain the token ids
        mode (str): mode of the passage format, coule be a list (processed) or a long string (unprocessed).
        kw (str) : the key word map to the passage in each data dictionary. Defaults to 'abstract'
        p_key (str) : the key word to search for specific passage. Default to 'title'
        model1 (nn.DataParallel) : local dependency encoder
        device (torch.device) : The device which models and data are on.
        model2 (nn.Module) : global coherence encoder
        model3 (nn.Module) : attention decoder
        
    Returns:
        result_list (list) : the list that contains the result of all samples organized in a dictionary form
        {taus, accs, pmrs, rouge-ws, pred, truth}
        over_all (dict) : the overall result of the result. keys include four metrics
    '''
    
    with torch.no_grad():
        print('test..............')

        valid_critic_dict = {'rouge-w' : rouge_w, 'acc' : acc, 'ken-tau' : kendall_tau, 'pmr' : pmr}

        result_list = []
        over_all = { 'Kendall-tau' : None, 'Accuracy' : None, 'ROUGE-w' : None, 'PMR' : None}

        accs = []
        rouge_ws = []
        ken_taus = []
        pmrs = []

        for superbatch in load_superbatch(testing_data_file, super_batch_size):

            bundles = []
            

            for data in superbatch:
                try:
                    bundles.append(convert_passage_to_samples_bundle(tokenizer, data, mode, kw, p_key))
                except:
                    traceback.print_exc()

            num_batch, valid_dataloader = homebrew_data_loader(bundles, batch_size=1)
            
            valid_value = []
            for step, batch in enumerate(valid_dataloader):
                try:
                    batch = tuple(t.to(device) for idx,t in enumerate(batch) if idx < 3)
                    pointers_output, ground_truth \
                        = dev_test(batch, model1, model2, model3, device)
                    # valid_value.append(valid_critic_dict[valid_critic](pointers_output, ground_truth))

                except Exception as err:
                    traceback.print_exc()
            
                rouge_ws.append(rouge_w(pointers_output, ground_truth))
                accs.append(acc(pointers_output, ground_truth))
                ken_taus.append(kendall_tau(pointers_output, ground_truth))
                pmrs.append(pmr(pointers_output, ground_truth))

                result_list.append({ 'Kendall-tau' : ken_taus[-1], 'Accuracy' : accs[-1], 'ROUGE-w' : rouge_ws[-1], 
                'PMR' : pmrs[-1], 'true': ground_truth, 'pred' : pointers_output})

            print('finishe {} samples. \n'.format(len(rouge_ws)))

        over_all['Kendall-tau'] = np.mean(ken_taus)
        over_all['Accuracy'] = np.mean(accs)
        over_all['ROUGE-w'] = np.mean(rouge_ws)
        over_all['PMR'] = np.mean(pmrs)
        
        print('Final scores:  kendall:{:.4f}, accuracy:{:.4f}, rouge-w:{:.4f}, pmr:{:.4f}\n'.format( \
            over_all['Kendall-tau'], over_all['Accuracy'], over_all['ROUGE-w'], over_all['PMR']))
    
    return result_list, over_all

def main(trained_model_file = './models/bert-base-cased.bin', test_data_file = './data/VALID_DATA_NAME', \
    out_json_file = './test/TEST_DUMP.JSON', super_batch_size = 200, mode = 'list', kw = 'abstract', p_key = 'title'):
    BERT_MODEL = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=False)
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    
    print("Loading model from {}".format(trained_model_file))
    model_state_dict = torch.load(trained_model_file)
    model1 = HierarchicalSentenceEncoder.from_pretrained(BERT_MODEL, state_dict=model_state_dict['params1'])
    model2 = GlobalGraphPropagation(model1.config.hidden_size)
    model2.load_state_dict(model_state_dict['params2'])
    model3 = BackwardFowardAttentiveDeoder(model1.config.hidden_size)
    model3.load_state_dict(model_state_dict['params3'])

    model1.to(device).eval()
    model2.to(device).eval()
    model3.to(device).eval()

    print('Start Training... on {} GPUs'.format(torch.cuda.device_count()))
    # model1 = torch.nn.DataParallel(model1, device_ids = range(torch.cuda.device_count()))
    result_pack, over_all = test(test_data_file, super_batch_size, tokenizer, mode, kw, p_key, \
        model1=model1, device=device, model2=model2, model3=model3) # stores the overall result into the over_all dictionary

    # we plan to output the attention prob in the future. Here we tend to simply 
    with open(out_json_file, 'w') as f_out:
        f_out.write(json.dumps(over_all) + ',\n') 
        for result_doc in result_pack:
            f_out.write(result_doc + ',\n') # we use the dictionary format to store the result of each sampel into the json dump file

    print ("Done.")
#import fire
if __name__ == "__main__":
    #fire.Fire(main)
    main()