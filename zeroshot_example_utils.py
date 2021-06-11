'''

ZS example utils

'''

import torch
import torch.nn.functional as F
import os
from transformers import GPT2Tokenizer



####
# This function takes a list of tokenized inputs (contexts) and targets
# and returns the list of - log P(target|context)
# either tokenwise or the sum, depending on 'red'
def get_nll_list(model, contexts, targets, batch=False, device=0, red=True):

    with torch.no_grad():
        torch.cuda.empty_cache()
        device = model.transformer.wte.weight.device

        if batch:
            nll_list = []
            while len(contexts)>0:
                nll_list += get_nll_list(model, contexts[:batch], targets[:batch], batch = False, red=red)
                contexts = contexts[batch:]
                targets = targets[batch:]
            return nll_list

        n_seqs = len(contexts)
        max_len = max([len(context + target) for context,target in zip(contexts, targets)])


        # initialize inputs
        inp = torch.zeros((n_seqs, max_len)).long().to(device)
        target_mask = torch.zeros((n_seqs, max_len))

        # construct input matrix
        for i in range(n_seqs):
            context, target = contexts[i], targets[i]

            # set values in input for this sequence
            inp[i, :len(context)] = torch.tensor(context).long()
            inp[i, len(context): len(context + target)] = torch.tensor(target).long()

            # mask is 1 where inp is the target
            target_mask[i, len(context): len(context +target)] = 1

        logits = model(inp.to(device))[0]
        #print(logits.shape)

        logits = logits[:,:-1].contiguous()
        logits_shape = logits.shape

        targ = inp[:,1:].contiguous()
        targ_mask = target_mask[:,1:]
        nll_mat = F.cross_entropy(logits.view(-1, logits.size(-1)), targ.view(-1),reduction='none') #CE_nored(logits.cpu().contiguous().view(-1, logits_shape[-1]), targ.view(-1))
        #ce_mat = F.cross_entropy(logits.contiguous().view(-1, logits_shape[-1]), targ.view(-1),reduction='none') #CE_nored(logits.cpu().contiguous().view(-1, logits_shape[-1]), targ.view(-1))
        nll_mat = nll_mat.cpu().view(logits_shape[:-1])
        if red:
            nll_list = (nll_mat*targ_mask).sum(dim=1)
            return nll_list.cpu().tolist()
        else:
            out_list = []
            for i in range(n_seqs):
                out_list += [nll_mat[i, targ_mask[i,:] == 1].tolist()]
            return out_list




gpt2_tokenizer_zs = GPT2Tokenizer.from_pretrained('gpt2')

####
# function that turns strings into tokenized example for zero-shot gpt-2
# 
def tokenize_for_gpt2_zs(grounding_str,context_str,target_str, return_all = False, trim_order = 'longest'):
    encoded_example = gpt2_tokenizer_zs.encode('{}\n{}\n{}'.format(grounding_str,context_str,target_str))

    precontext = gpt2_tokenizer_zs.encode('{}\n'.format(grounding_str))
    context = gpt2_tokenizer_zs.encode('{}\n'.format(context_str))
    target = gpt2_tokenizer_zs.encode('{}\n'.format(target_str))
    

    inp = precontext + context
    
    # if doing context first, first try trimming the context, before touching the grounding
    if trim_order == 'context_first':
        while len(inp + target) > 1024 and len(context) > 1:
            context = context[1:]
            inp = precontext + context
    
    while len(inp + target) > 1024:
        if len(precontext) > len(context):
            precontext = precontext[:-1]
        else:
            context = context[1:]
        inp = precontext + context
        
                                      
    if return_all:
        return precontext, context, target                   
    return inp, target


## This truncates input strings for zero-shot gpt2
## so that they fit in the context window
def get_truncated_inputs_gpt2_zs(grounding_str, context_str, target_str, tokenizer, trim_order = 'shortest'):
    precontext, context, target = tokenize_for_gpt2_zs(grounding_str,context_str,target_str, return_all = True)

    inp = precontext + context
    
    # if doing context first, first try trimming the context, before touching the grounding
    if trim_order == 'context_first':
        while len(inp + target) > 1023 and len(context) > 0:
            context = context[1:]
            inp = precontext + context
    
    while len(inp + target) > 1023:
        if len(precontext) > len(context):
            precontext = precontext[:-1]
        else:
            context = context[1:]
            
        inp = precontext + context
    # return the trimmed grounding and context without any special tokens (Trim 
    # the first token of precontext, and last token of context, which are special tokens)
    return tokenizer.decode(precontext), tokenizer.decode(context)