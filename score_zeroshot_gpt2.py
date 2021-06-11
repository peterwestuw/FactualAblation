from FA_data_utils import get_FA_synthetic_dataset, get_FA_wiki_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
from zeroshot_example_utils import get_nll_list, tokenize_for_gpt2_zs, get_truncated_inputs_gpt2_zs
import math

###
# Load Factual Ablation datasets
###
FA_wiki_dataset = get_FA_wiki_dataset()
FA_synth_dataset = get_FA_synthetic_dataset()

###
# Load zero-shot gpt2 and tokenizer
###
gpt2_tokenizer_zs = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').eval().cuda()







####
# Evaluate on the synthetic dataset (identical to above besides dataset and margin)
####
print('='*20)
print('Evaluating on synthetic dataset...')
print('='*20)
model_scores = []

## get scores for true and false grounding over the full dataset
for ex in tqdm(FA_synth_dataset):
    
    ## get the components of the example; each is a string
    grounding_True, grounding_False, context, target_str = ex
    
    ## First, get the target score given the True grounding
    # truncate inputs to fit in the model window
    gnd, ctxt = get_truncated_inputs_gpt2_zs(grounding_True, context, target_str, gpt2_tokenizer_zs, trim_order = 'shortest')
    # tokenize inputs
    inp, target = tokenize_for_gpt2_zs(gnd,ctxt,target_str)
    # get the nll of the target under true grounding
    score_True = get_nll_list(model,[inp],[target])[0]

    ## Next, get the target score given the False grounding (same as above)
    gnd, ctxt = get_truncated_inputs_gpt2_zs(grounding_False, context, target_str, gpt2_tokenizer_zs, trim_order = 'shortest')
    inp, target = tokenize_for_gpt2_zs(gnd,ctxt,target_str)
    score_False = get_nll_list(model,[inp],[target])[0]
    
    model_scores.append((score_True,score_False))
    
## get accuracy (margin=0) and margin-accuracy
print('='*20)
margin= 0 
print('margin_acc (m = {}): {}'.format(margin, sum([(v[1] - v[0] > margin) for v in model_scores] ) / len(model_scores)))
margin= math.log(100)
print('margin_acc (m = {}): {}'.format(margin, sum([(v[1] - v[0] > margin) for v in model_scores] ) / len(model_scores)))
print('='*20)



####
# Evaluate on the wiki dataset (identical to above besides dataset and margin)
####
print('='*20)
print('Evaluating on natural (wiki) dataset...')
print('='*20)
model_scores = []
## get scores for true and false grounding over the full dataset
for ex in tqdm(FA_wiki_dataset):
    
    ## get the components of the example; each is a string
    grounding_True, grounding_False, context, target_str = ex
    
    ## First, get the target score given the True grounding
    # truncate inputs to fit in the model window
    gnd, ctxt = get_truncated_inputs_gpt2_zs(grounding_True, context, target_str, gpt2_tokenizer_zs, trim_order = 'shortest')
    # tokenize inputs
    inp, target = tokenize_for_gpt2_zs(gnd,ctxt,target_str)
    # get the nll of the target under true grounding
    score_True = get_nll_list(model,[inp],[target])[0]

    ## Next, get the target score given the False grounding (same as above)
    gnd, ctxt = get_truncated_inputs_gpt2_zs(grounding_False, context, target_str, gpt2_tokenizer_zs, trim_order = 'shortest')
    inp, target = tokenize_for_gpt2_zs(gnd,ctxt,target_str)
    score_False = get_nll_list(model,[inp],[target])[0]
    
    model_scores.append((score_True,score_False))
    
## get accuracy (margin=0) and margin-accuracy
print('='*20)
margin= 0 
print('margin_acc (m = {}): {}'.format(margin, sum([(v[1] - v[0] > margin) for v in model_scores] ) / len(model_scores)))
margin= math.log(1000)
print('margin_acc (m = {}): {}'.format(margin, sum([(v[1] - v[0] > margin) for v in model_scores] ) / len(model_scores)))
print('='*20)