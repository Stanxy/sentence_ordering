# This file is the main process of the whole pipeline
import json
import os
from tqdm import tqdm, trange
from traceback import print_exc
import numpy as np
# waiting to be finished
from data import convert_passage_to_samples_bundle, load_superbatch, homebrew_data_loader
from model import HierarchicalSentenceEncoder, GlobalGraphPropagation, BackwardFowardAttentiveDecoder
from model import calculate_loss, dev_test
from utils import WindowMean, acc, kendall_tau, pmr, rouge_w, warmup_linear

from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import torch
from torch import nn
from torch.optim import Adam, Adadelta
import traceback

def train(training_data_file, valid_data_file, super_batch_size, tokenizer, mode, kw, p_key, model1, device, model2, model3, \
            batch_size, num_epoch, gradient_accumulation_steps, lr1, lr2, lambda_, valid_critic, early_stop):
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
        device (torch.device): The device which models and data are on.
        model2 (nn.Module): global coherence encoder
        model3 (nn.Module): attention decoder
        batch_size (int): Defaults to 4.
        num_epoch (int): Defaults to 1.
        gradient_accumulation_steps (int): Defaults to 1. 
        lr (float): Defaults to 1e-4. The Start learning rate.
        lambda_ (float): Defaults to 0.01. Balance factor for param nomalization.
        valid_critic (bool) : what critic to use when early stop evaluation. Default to 5 
        early_stop (int) : set the early stop boundary. Default to 5 

    '''

    # Prepare optimizer for Sys1
    param_optimizer_bert = list(model1.named_parameters())
    param_optimizer_others = list(model2.named_parameters()) + list(model3.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight'] 
    # We tend to fix the embedding. Temeporarily we doesn't find the embedding layer
    optimizer_grouped_parameters_bert = [
        {'params': [p for n, p in param_optimizer_bert if not any(nd in n for nd in no_decay)], 'weight_decay': lambda_},
        {'params': [p for n, p in param_optimizer_bert if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer_grouped_parameters_others = [
        {'params': [p for n, p in param_optimizer_others if not any(nd in n for nd in no_decay)], 'weight_decay': lambda_},
        {'params': [p for n, p in param_optimizer_others if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    critic = nn.NLLLoss(reduction='none')

    line_num = int(os.popen("wc -l " + training_data_file).read().split()[0])
    global_step = 0 # global step
    opt1 = BertAdam(optimizer_grouped_parameters_bert, lr = lr1, warmup = 0.1, t_total=line_num / batch_size * num_epoch) # optimizer 1
    # opt = Adam(optimizer_grouped_parameter, lr=lr)
    opt2 = Adadelta(optimizer_grouped_parameters_others, lr = lr2, rho=0.95)
    model1.to(device) # 
    model1.train() # 
    model2.to(device) # 
    model2.train() # 
    model3.to(device) # 
    model3.train() # 

    for epoch in trange(num_epoch, desc = 'Epoch'):
        
        smooth_mean = WindowMean()
        opt1.zero_grad()
        opt2.zero_grad()

        for superbatch, line_num in load_superbatch(training_data_file, super_batch_size):
            bundles = []
            
            for data in superbatch:
                try:
                    bundles.append(convert_passage_to_samples_bundle(tokenizer, data, mode, kw, p_key))
                    
                except:
                    print_exc()
            
            num_batch, dataloader = homebrew_data_loader(bundles, batch_size=batch_size)
            
            tqdm_obj = tqdm(dataloader, total = num_batch)
            num_steps = line_num #
            for step, batch in enumerate(tqdm_obj):
                try:
                    #batch[0] = batch[0].to(device)
                    #batch[1] = batch[1].to(device)
                    #batch[2] = batch[2].to(device)
                    batch = tuple(t for t in enumerate(batch))
                    log_prob_loss, pointers_output, ground_truth = calculate_loss(batch, model1, model2, model3, device, critic)
                    # here we need to add code to cal rouge-w and acc
                    rouge_ws = rouge_w(pointers_output, ground_truth)
                    accs = acc(pointers_output, ground_truth)
                    ken_taus = kendall_tau(pointers_output, ground_truth)
                    pmrs = pmr(pointers_output, ground_truth)

                    log_prob_loss.backward()

                    # ******** In the following code we gonna edit it and made early stop ************

                    if (step + 1) % gradient_accumulation_steps == 0:
                        # modify learning rate with special warm up BERT uses. From BERT pytorch examples
                        lr_this_step = lr1 * warmup_linear(global_step / num_steps, warmup = 0.1)
                        for param_group in opt1.param_groups:
                            param_group['lr'] = lr_this_step
                        global_step += 1
                        
                        opt2.step()
                        opt2.zero_grad()
                        smooth_mean_loss = smooth_mean.update(log_prob_loss.item())
                        tqdm_obj.set_description('{}: {:.4f}, {}: {:.4f}, smooth_mean_loss: {:.4f}'.format(
                            'accuracy', accs, 'rough-w', rouge_ws,  smooth_mean_loss))
                        # During warming period, model1 is frozen and model2 is trained to normal weights
                        if smooth_mean_loss < 1.0 and step > 100: # ugly manual hyperparam
                            warmed = True
                        if warmed:
                            opt1.step()
                        opt1.zero_grad()
                        if step % 1000 == 0:
                            output_model_file = './models/bert-base-cased.bin.tmp'
                            saved_dict = {'params1' : model1.module.state_dict()}
                            saved_dict['params2'] = model2.state_dict()
                            saved_dict['params3'] = model3.state_dict()
                            torch.save(saved_dict, output_model_file)

                except Exception as err:
                    traceback.print_exc()
                    # if mode == 'list':   
                    #     print(batch._id) 

        if epoch < 5:
            best_score = 0
            continue

        with torch.no_grad():
            print('valid..............')

            valid_critic_dict = {'rouge-w' : rouge_w, 'acc' : acc, 'ken-tau' : kendall_tau, 'pmr' : pmr}

            for superbatch, _ in load_superbatch(valid_data_file, super_batch_size):
                bundles = []
                
                for data in superbatch:
                    try:
                        bundles.append(convert_passage_to_samples_bundle(tokenizer, data, mode, kw, p_key))
                    except:
                        print_exc()

                num_batch, valid_dataloader = homebrew_data_loader(bundles, batch_size=1)
                
                valid_value = []
                for step, batch in enumerate(valid_dataloader):
                    try:
                        batch = tuple(t for idx,t in enumerate(batch) )
                        pointers_output, ground_truth \
                            = dev_test(batch, model1, model2, model3, device)
                        valid_value.append(valid_critic_dict[valid_critic](pointers_output, ground_truth))

                    except Exception as err:
                        traceback.print_exc()
                        # if mode == 'list':   
                        #     print(batch._id) 

                score = np.mean(valid_value)
            print('epc:{}, {} : {:.2f} best : {:.2f}\n'.format(epoch, valid_critic, score, best_score))

            if score > best_score:
                best_score = score
                best_iter = epoch

                print('Saving model to {}'.format(output_model_file)) # save model structure
                saved_dict = {'params1' : model1.module.state_dict()} # save parameters
                saved_dict['params2'] = model2.state_dict() # save parameters
                saved_dict['params3'] = model3.state_dict()
                torch.save(saved_dict, output_model_file) #

                # print('save best model at epc={}'.format(epc))
                # checkpoint = {'model': model.state_dict(),
                #             'args': args,
                #             'loss': best_score}
                # torch.save(checkpoint, '{}/{}.best.pt'.format(args.model_path, args.model))


            if early_stop and (epoch - best_iter) >= early_stop:
                print('early stop at epc {}'.format(epoch))
                break

def main(output_model_file = './models/bert-base-cased.bin', training_data_file = './data/TRAIN_DATA_NAME', \
        valid_data_file = './data/VALID_DATA_NAME', super_batch_size = 200, mode = 'list', kw = 'abstract', \
        batch_size = 4, num_epoch = 1, gradient_accumulation_steps = 1, lr1 = 1e-4, lr2 = 1e-4, \
        lambda_ = 0.01, p_key = 'title', valid_critic = 'ken-tau', early_stop = 5, load=False):
    BERT_MODEL = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=False)
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    if load: # load pretrained version
        # Here if we have pretrained model we could use this
        print("Loading model from {}".format(output_model_file))
        model_state_dict = torch.load(output_model_file)
        model1 = HierarchicalSentenceEncoder.from_pretrained(BERT_MODEL, state_dict=model_state_dict['params1'])
        model2 = GlobalGraphPropagation(model1.config.hidden_size)
        model2.load_state_dict(model_state_dict['params2'])
        model3 = BackwardFowardAttentiveDecoder(model1.config.hidden_size)
        model2.load_state_dict(model_state_dict['params3'])

    else: # build new model
        # build sentence_encoder
        model1 = HierarchicalSentenceEncoder.from_pretrained(BERT_MODEL,
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1))
        print ("Finish building model1")
        model2 = GlobalGraphPropagation(model1.config.hidden_size)
        print ("FInish building model2")
        model3 = BackwardFowardAttentiveDecoder(model1.config.hidden_size)
        print ("FInish building model2")
    
    print('Start Training... on {} GPUs'.format(torch.cuda.device_count()))
    model1 = torch.nn.DataParallel(model1, device_ids = range(torch.cuda.device_count()))
    train(training_data_file, super_batch_size, tokenizer, mode, kw, p_key, \
        model1=model1, device=device, model2=model2, model3=model3, # Then pass hyperparams
        batch_size=batch_size, num_epoch=num_epoch, gradient_accumulation_steps=gradient_accumulation_steps, \
            lr11=lr1, lr2=lr2, lambda_=lambda_, valid_critic = valid_critic, early_stop = early_stop) 

#import fire
if __name__ == "__main__":
    #fire.Fire(main)
    main()