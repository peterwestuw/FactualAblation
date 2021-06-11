'''

This is a first script to post-process json files returned by gerold's code
mainly, clean the data and put it in the same format as Content Transfer dataset

'''


import csv
import json
import ast
import re
import sys
import nltk
import mwparserfromhell
import os, io, argparse

#csv.field_size_limit(sys.maxsize)







###################################### Functions

'''
This is new code that replaces refs first...

'''
 
def compress_whitespace(s):
    return re.sub("[ \t]+"," ",s)

def scrub_refs(s):
    s = str(mwparserfromhell.parse(s).strip_code(keep_template_params=True))
    
    ref_regex = re.compile(r"<ref.*?</ref>")

    ref_indices = [(m.start(0), m.end(0)) for m in ref_regex.finditer(s)]

    ref_dict = {}
    ref_key = []
    for (i,j) in ref_indices:
        ref = s[i:j]
        key = hash(ref)
        
        ref_key += [(ref,key)]
    
    for ref,key in ref_key:
        s = s.replace(ref, ' <|REF|{}|> '.format(key))
        ref_dict[key] = ref
        
    # drop any resigual tags
    tag_regex = re.compile(r"<ref.*>", flags=re.MULTILINE)
    s = tag_regex.sub('', s)
    tag_regex = re.compile(r"<br.*>", flags=re.MULTILINE)
    s = tag_regex.sub('', s)
    
    return s, ref_dict

def clean_sent(s):
    
    s = s.replace('[', '')
    s = s.replace(']', '')
    
    toks = s.split(' ')
    
    for i, tok in enumerate(toks):
        subtoks = tok.split('\n')
        
        if len(subtoks) > 1:
            subtoks =  [ t for t in subtoks if bool(re.search('[a-zA-Z0-9\']+', t))]
        
        toks[i] = (' '.join(subtoks)).strip()
        
    #toks = [t for t in toks if not ('\n' in t and not bool(re.search('[a-zA-Z0-9]+', t)))]
    #toks = [t.replace('\n','') for t in toks]
    
    
    s = (' '.join(toks)).strip()
    
    return s

def get_diff_ind(original, edited, whitespace = False, sentence = False):
    
    if sentence:
        sent_list_original = nltk.sent_tokenize(original)
        sent_list_edited = nltk.sent_tokenize(edited)
        
        for j in range(min([len(sent_list_original), len(sent_list_edited)])):
            ## first, remove references 
            ref_regex = re.compile(r"<\|REF\|.*?\|>")
            s_original = ref_regex.sub('', sent_list_original[j])
            s_edited = ref_regex.sub('', sent_list_edited[j])
            
            # clear extra whitespace
            s_original = (' '.join(s_original.split(' '))).strip()
            s_edited = (' '.join(s_edited.split(' '))).strip()
            
            # check if this is where they deviate
            if s_original != s_edited:
                break
                
                

        # push i forward until we include all of the sentence prefix
        i = sum([len(s) for s in sent_list_edited[:j]])

             
        while not all([s in edited[:i] for s in sent_list_edited[:j]]):
            i += 1
            #print(all([s in edited[:i] for s in sent_list_edited[:j]]))
        return i
    
    
    common_prefix = ''

    for i in range(min([len(edited), len(original)])):
        if edited[i] != original[i]:
            break
        common_prefix = common_prefix + edited[i]

    # if 'whitespace', go to last whitespace within the common prefix
    # UNLESS the next added character is whitespace (then we are at the 
    # end of a token/word)
    if whitespace and edited[i] not in ['\n', ' ']:
        ws_regex = re.compile(r"[\s\n]")
        indices = [(m.start(0), m.end(0)) for m in ws_regex.finditer(common_prefix)]
        return indices[-1][1]

    return i


# this function scrubs the prefix of formatting
# once it's cleaned, should join it with cleaned edit,
# and then sentence tokenize and take the first sentence that doesn't match 
# the prefix
def clean_prefix(s, k = 3):
    s = str(mwparserfromhell.parse(s).strip_code(keep_template_params=True))
    
    ref_regex = re.compile(r"<\|REF\|.*?\|>", flags=re.MULTILINE)

    s = ref_regex.sub('', s)
    
    # drop any resigual tags
    tag_regex = re.compile(r"<ref.*>", flags=re.MULTILINE)
    s = tag_regex.sub('', s)
    tag_regex = re.compile(r"<br.*>", flags=re.MULTILINE)
    s = tag_regex.sub('', s)
    
    s = clean_sent(s)

    sent_list = nltk.tokenize.sent_tokenize(s)
    
    if len(sent_list) <= 3:
        return ' '.join(sent_list)
    else:
        return ' '.join(sent_list[-3:])    
    return s



'''
Plan for cleaning the edit:

First, replace every ref, then sentence tokenize. 

Only considering the first sentence, grab all refs included in the sentence,

And then drop the suffix (any other text) and the references, only saving the urls

FOR REF (it may appear in the FIRST word of the next sentence, so check that too...)

'''

def clean_target(s, ref_dict, whitelist = None):
    #s = str(mwparserfromhell.parse(s).strip_code(keep_template_params=True))
    

    # drop any resigual tags
    tag_regex = re.compile(r"<ref.*>", flags=re.MULTILINE)
    s = tag_regex.sub('', s)
    tag_regex = re.compile(r"<br.*>", flags=re.MULTILINE)
    s = tag_regex.sub('', s)
    
    s = clean_sent(s)

    
    #s, refs = clean_target(edited[i:])
    
    sent_list = nltk.tokenize.sent_tokenize(s)

    # if the first sentence is trivial, combine it with the next until it's long enough...
    while len(sent_list[0]) < 10 and len(sent_list) > 1:
        sent_list = [sent_list[0] + sent_list[1]] + sent_list[1:]
    
    # get the tags of all refs to include
    ref_tags = []
    

    ref_regex = re.compile(r"<\|REF\|.*?\|>")
    ref_tag_indices = [(m.start(0), m.end(0)) for m in ref_regex.finditer(sent_list[0])]
    
    
    
    for (i,j) in ref_tag_indices:
        ref_tags += [sent_list[0][i:j].split('|')[2]]

    ## check if the first token of the next sentence is <|REF|>
    if len(sent_list)>1 and sent_list[1].startswith('<|REF'):        
        # get the tag for the first ref in the next sentence if it starts with a ref...
        ref_tags += [ref_regex.findall(s)[0].split('|')[2]]
        
    refs = [ref_dict[int(tag)] for tag in ref_tags]
        

    ## get the urls from the first k refs: 
    url_regex = re.compile(r"https?://[^\s|\]]+", flags=re.MULTILINE)

    urls = []
    for ref in refs:
        urls += url_regex.findall(ref)



    ## process the url, replacing any bad formatting etc
    urls = [url.replace('</ref>','') for url in urls]



    ## filter urls for domain (keep all for now, in case only 1 is accessible by common crawl)
    if whitelist is not None:
        urls = [url for url in urls if any([d in url for d in whitelist] ) ]
    
    # clear out refs
    s_out =  ref_regex.sub(' ',sent_list[0])
    # remove extra spaces
    s_out = (' '.join(s_out.split(' '))).strip()
    
    return ref_regex.sub(' ',sent_list[0]), urls

## define filter conditions

## as long as this is idempotent, we can expand the definition and run further filtering
## steps with the same function. The key is to winnow the options down at the beginning, small 
# enough that everything fits in one file...
def valid_data(d):
    valid = True
    
    # both the edited and original have valid references
    valid = valid and (len(d['urls_edit']) > 0 or len(d['urls_orig']) > 0)
    
    # the references are not the same before and after edits
    valid = valid and (str(d['urls_edit']) != str(d['urls_orig']))
    
    # !!! add one for news domains (at least a loose one...)
    
    
    # less than 100 words
    valid = valid and len(d['target_edit'].split()) < 100
    # less than 100 words
    valid = valid and len(d['target_orig'].split()) < 100
    
    
    # !!! add one for length 
    
    return valid

######################################


from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int, help='the index of the file (within the dump status file) to download and process')
    parser.add_argument('--input-path', type=str, default="./data/raw/", help='the temp / data directory, used to download wiki dump files')
    parser.add_argument('--output-path', type=str, default="./data/out/", help='the output directory')


    parser.add_argument('--azure', action='store_true')
    args = parser.parse_args()
    
    in_file = args.input_path + '{}.json'.format(args.index)
    out_file = args.output_path + '{}_proc1.json'.format(args.index)



    with open(in_file,'r', encoding='utf8') as f:
        lines = f.readlines()

    data = []
    for line in tqdm(lines):
        try:
            line = json.loads(line, )


            original, ref_dict = scrub_refs(line['original_src_text'])
            edited, ref_dict_ = scrub_refs(line['original_tgt_text'])

            ref_dict.update(ref_dict_)


            i = get_diff_ind(original, edited, sentence=True)

            prefix = clean_prefix(edited[:i])
            target_1, urls_1 = clean_target(edited[i:], ref_dict, whitelist = None)

            try:
                target_2, urls_2 = clean_target(original[i:], ref_dict, whitelist = None)
            except:
                target_2, urls_2 = '',[]

        except:
            print('Fail')

        datum_out = {'context':compress_whitespace(prefix).strip(), 
                  'target_edit':compress_whitespace(target_1).strip(),
                  'urls_edit':urls_1, 
                  'target_orig':compress_whitespace(target_2).strip(), 
                  'urls_orig':urls_2, 'og_edit':edited[i:i+600], 'og_orig':original[i:i+600]}

        preserve_keys = ['rev_id','page_id','parent_id','timestamp','diff_url']

        for key in preserve_keys:# these are keys to save from the original json
            datum_out[key] = line[key]

        data += [datum_out]




    data_ = data

    data_out = [d for d in data_ if valid_data(d)]



    with open(out_file,'w') as f:

        for d in data_out:
            s = json.dumps(d) + '\n'
            f.write(s)
