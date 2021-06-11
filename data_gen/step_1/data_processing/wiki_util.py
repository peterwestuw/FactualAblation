import difflib
import glob
import html
import random
import re
import bz2
import io
import codecs
import collections
import logging

import numpy as np
import mwparserfromhell

## Re-enable when using spacy!
#import spacy
#nlp = spacy.load('en_core_web_sm') # was: 'en'

from tqdm import tqdm
import nltk
nltk.data.path.append('./nltk_data/')
from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize

RevisionMETA = collections.namedtuple("RevisionMETA", ['comment_text',
    'rev_id', 'parent_id', 'text_length', 'section_title', 'page_title'])

'''
Extract the contents by delimitors
'''
def extract_with_delims(content, start_delim, end_delim, start_idx):
    delims_start = content.find(start_delim, start_idx)
    if delims_start == -1:
        return '', start_idx

    delims_end = content.find(end_delim, start_idx)
    if delims_end == -1:
        return '', start_idx

    if delims_end <= delims_start:
        return '', start_idx

    delims_start += len(start_delim)

    return content[delims_start:delims_end], delims_end

'''
Extract the contents of revisions, e.g., revision_id, parent_id, user_name, comment, text
'''
def extract_data(revision_part):
    rev_id, next_idx = extract_with_delims(revision_part, "<id>", "</id>", 0)
    parent_id, next_idx = extract_with_delims(revision_part, "<parentid>", "</parentid>", next_idx)
    timestamp, next_idx = extract_with_delims(revision_part, "<timestamp>", "</timestamp>", next_idx)
    username, next_idx = extract_with_delims(revision_part, "<username>", "</username>", next_idx)
    userid, next_idx = extract_with_delims(revision_part, "<id>", "</id>", next_idx)
    # For annoymous user, the ip address will be used instead of the user name and id
    userip, next_idx = extract_with_delims(revision_part, "<ip>", "</ip>", next_idx)
    comment, next_idx = extract_with_delims(revision_part, "<comment>", "</comment>", next_idx)
    text, next_idx = extract_with_delims(revision_part, "<text xml:space=\"preserve\">", "</text>", next_idx)
    return (rev_id, parent_id, timestamp, username, userid, userip, comment, text)


def split_pages(wiki_file, chunk_size=1024):

    text_buffer = ""
    cur_index = 0
    cursor = 0
    page_start_index = -1
    page_end_index = -1

    while True:
        try:
            chunk = wiki_file.read(chunk_size)
           # print('split_pages l65')

            if chunk:
                text_buffer += chunk
        except MemoryError:
            # if memory error, abandon current buffer and restart
            text_buffer = ""
            page_start_index = -1
            page_end_index = -1
            cursor = 0
            cur_index = 0
            continue

        #print(len(text_buffer))
        cur_index = 0
        PAGE_START = "<page>"
        PAGE_END = "</page>"
        
        while True:
            if page_start_index == -1:
                page_start_index = text_buffer.find(PAGE_START, cursor)
                if page_start_index == -1:
                    cursor = len(text_buffer) - 1 # set cursor to last index
                    break
                cursor = page_start_index + len(PAGE_START)
           
            if page_end_index == -1:
                page_end_index = text_buffer.find(PAGE_END, cursor)
                if page_end_index == -1:
                    cursor = len(text_buffer) - 1
                    break
                cursor = page_end_index + len(PAGE_END)

            yield text_buffer[page_start_index:page_end_index + len(PAGE_END)]

            cur_index = page_end_index + len(PAGE_END)
            page_start_index = -1
            page_end_index = -1

        if chunk == "":
            break

        if cur_index == -1:
            text_buffer = ""
        else:
            text_buffer = text_buffer[cur_index:]
            cursor -= cur_index
            if page_start_index != -1:
                page_start_index -= cur_index 
            if page_end_index != -1:
                page_end_index -= cur_index


'''
Extract the revision text buffer, which has the format "<revision> ... </revision>".
'''
def split_records(wiki_file, azure=False, chunk_size=150 * 1024, max_bytes=None):
    
    text_buffer = ""    
    cur_index = 0
    if azure:
        blob_size = wiki_file.get_blob_properties()['size']
        cur_offset = 0 

        decompressor = bz2.BZ2Decompressor()
        decoder = codecs.getincrementaldecoder('utf-8')()

    page_title = ""
    page_id = ""
    
    next_page_start = None
    next_page_title_id = None

    total_bytes_read = 0
    while True:
        if not azure:
            chunk = wiki_file.read(chunk_size)
            total_bytes_read += chunk_size
        else:
            if cur_offset < blob_size:
                bytes_data = wiki_file.download_blob(offset=cur_offset,
                    length=chunk_size).content_as_bytes()
                cur_offset += len(bytes_data)
                chunk = decompressor.decompress(bytes_data)
                chunk = decoder.decode(input=chunk)
                total_bytes_read += chunk_size
            else:
                chunk = ''
                
        if chunk:
            text_buffer += chunk

        cur_index = 0
        REVISION_START = "<revision>"
        REVISION_END = "</revision>"
        PAGE_START = "<page>"
        PAGE_TITLE_START = "<title>"
        PAGE_TITLE_END = "</title>"
        
        while True:

            if next_page_start is None:
                # we did not yet find where the next page starts, keep looking for it..
                page_start_index = text_buffer.find(PAGE_START, cur_index)
                found_new_page_start_in_buffer = page_start_index != -1
                if found_new_page_start_in_buffer:
                    # we found the start of the next page, store it so that we notice once a revision passes over it
                    next_page_title, _ = extract_with_delims(text_buffer, PAGE_TITLE_START, PAGE_TITLE_END, page_start_index)
                    next_page_id, _ = extract_with_delims(text_buffer, "<id>", "</id>", page_start_index)
                    
                    # there is a chance that the page title and id are not yet in this buffer/chunk. only set of we could parse them correctly
                    if next_page_title and next_page_id:
                        next_page_title_id = (next_page_title, next_page_id)
                        next_page_start = page_start_index

            # find the revision start position
            revision_start_index = text_buffer.find(REVISION_START, cur_index)

            passed_page = next_page_start is not None and revision_start_index > next_page_start
            if passed_page:
                # the current revision passed over the next pagebreak. update the current page_title, page_id and start looking for the next new page!
                page_title, page_id = next_page_title_id
                next_page_title_id = None
                next_page_start = None

            # No revision in the buffer, continue loading data
            if revision_start_index == -1:
                break

            # find the revision end position
            revision_end_index = text_buffer.find(REVISION_END, revision_start_index)

            # No complete page in buffer
            if revision_end_index == -1:
                break

            if next_page_start is not None and revision_end_index > next_page_start:
                logging.error("Error: the revision ends at {} but next page was scheduled to start at {}. This should never happen!".format(revision_end_index, next_page_start))

            if not page_title:
                logging.error("Error: missing page title. This should never happen!")

            revision_text = text_buffer[revision_start_index:revision_end_index + len(REVISION_END)]
            yield page_title, page_id, revision_text

            cur_index = revision_end_index + len(REVISION_END)

        if max_bytes is not None and total_bytes_read > max_bytes:
            logging.info("\nStop processing input stream as max_bytes={} was requested and already read a total of {} bytes\n".format(max_bytes, total_bytes_read))
            break

        # No more datA
        if chunk == "":
            break

        if cur_index == -1:
            text_buffer = ""
        else:
            text_buffer = text_buffer[cur_index:]
            if next_page_start is not None:
                next_page_start -= cur_index
                if next_page_start < 0:
                    next_page_start = None
                    logging.error("Error: the text buffer was clipped in a way that the page title was cut. This should never happen!")

def split_into_sections(text):
    section_title_pattern = '(^=+\s*.+\s*=+$)'
    section_splits = re.split(section_title_pattern, text, flags=re.MULTILINE)
    section_texts = section_splits[::2]
    section_titles = ['Lead'] + section_splits[1::2]
    section_titles = [t.strip('= ') for t in section_titles]

    return section_texts, section_titles

def is_contiguous(seq):
    '''
    Check if ordered (ascending) sequence of numbers is contiguous
    '''

    return (seq[-1] - seq[0]) == (len(seq) - 1)

def split_edits(source_text, target_text, k=5):
    source_sections, source_titles = split_into_sections(source_text)
    target_sections, target_titles = split_into_sections(target_text)

    tgt_sect_dict = {title: text for title,text in zip(target_titles,\
            target_sections)}

    splitter = SentenceSplitter(language='en')

    for src_sect, src_title in zip(source_sections, source_titles):
        if src_title in tgt_sect_dict.keys():
            tgt_sect = tgt_sect_dict[src_title]
        else:
            continue

        # retrieve references from text
        src_sect, src_references = retrieveReferences(src_sect)
        tgt_sect, tgt_references = retrieveReferences(tgt_sect)
        
        # strip text of markup using mwparserfromhell
        src_sect = str(mwparserfromhell.parse(src_sect).strip_code())
        tgt_sect = str(mwparserfromhell.parse(tgt_sect).strip_code())

        source_sentences = splitter.split(src_sect) 
        target_sentences = splitter.split(tgt_sect)

        # empty lines get split into empty sentences!
        source_sentences = [word_tokenize(s) for s in source_sentences if\
                len(s)]
        target_sentences = [word_tokenize(s) for s in target_sentences if\
                len(s)]

        for i, src_sent in enumerate(source_sentences):
            min_idx = max(i-k, 0)
            max_idx = min(i+k, len(target_sentences))
            if min_idx >= max_idx:
                continue
            bleu_scores = [sentence_bleu([src_sent], target_sentences[j])\
                    for j in range(min_idx, max_idx)]
            try:
                match_idx = np.argmax(bleu_scores) + min_idx
                match_score = np.max(bleu_scores)
            except ValueError as e:
                print(min_idx, max_idx, len(bleu_scores))
                raise e

            tgt_sent = target_sentences[match_idx]
            tgt_lctx = target_sentences[max(match_idx-k,0):match_idx]
            src_lctx = source_sentences[max(i-k, 0):i]

            source_diff, target_diff = diffRevision(src_sent, tgt_sent)
            if source_diff and not target_diff: #deletion
                #if is_contiguous(source_diff): yield src_sent, tgt_sent,\
                        #tgt_lctx
                continue
            elif not source_diff and target_diff:
                if is_contiguous(target_diff): yield src_sent, tgt_sent,\
                        src_lctx, tgt_lctx, source_diff, target_diff


def sampleNext(sample_ratio):
    return random.random() < sample_ratio

def cleanCmntText(comment):
    filter_words = []
    comment = comment.replace("(edited with [[User:ProveIt_GT|ProveIt]]", "")
    #comment = re.sub("(edited with \[\[User\:ProveIt_GT\|ProveIt\]\]", "", comment)
    return comment
    

def checkComment(comment, comment_tokens, min_comment_length):

    if len(comment_tokens) < min_comment_length:
        return False

    filter_words = ["[[Project:AWB|AWB]]", "[[Project:AutoWikiBrowser|AWB]]", "Undid revision"]
    if any(word in comment for word in filter_words):
        return False

    return True

'''
clean the wiki text
E.g. "[[link name]] a&quot; bds&quot; ''markup''" to "link name a bds markup"
'''
def cleanWikiText(wiki_text): # use mwparserfromhell instead
    '''
    Cleans wikipedia text and retrieves references
    '''

    # resolve [[Category|name]] links
    wiki_text = re.sub('\[\[[^\[\]]+\|([^\[\]]+)\]\]', r'\1', wiki_text) 
    
    # replace link: [[link_name]] and quotes
    wiki_text = re.sub("\[\[", "", wiki_text)
    wiki_text = re.sub("\]\]", "", wiki_text)
    wiki_text = re.sub("''", "", wiki_text)

    # replace '<', '>', '&'    
    # wiki_text = re.sub("&quot;", "", wiki_text)
    # wiki_text = re.sub("&lt;", "<", wiki_text)
    # wiki_text = re.sub("&gt;", ">", wiki_text)
    # wiki_text = re.sub("&amp;", "&", wiki_text)

    # use html unescape to decode the html special characters
    #wiki_text = html.unescape(wiki_text) # do outside of cleanText

    # remove markup
    html_elements = [
        '<table.*>.*</table>', #tables
        '<!--[^<>]*-->', # comments
        '<\w+>[^<>]*</\w+>',
        '<\w+>|<\w+/>', # html tags
        '{{[^{}]*}}', # wikipedia elements
        '&nbsp;'
    ]
    for pattern in html_elements:
        wiki_text = re.sub(pattern, '', wiki_text, flags=re.DOTALL)

    wiki_text = wiki_text.strip('\n')

    return wiki_text

def retrieveReferences(text):
    deref_text = ''
    prev_idx = 0
    reference_pattern = '<ref[^<>]*/>|<ref[^<>]*>[^<>]*</ref>|\{\{sfn\|[^{}]*\}\}'
    ref_counter = 0
    ref_list = []
    for match in re.finditer(reference_pattern, text):
        span = match.span()
        deref_text += text[prev_idx:span[0]]
        prev_idx = span[1]
        #deref_text += " (ref {}) ".format(ref_counter)
        ref_counter += 1
        ref_list.append(text[span[0]:span[1]])

    # if no references found, return original text
    if not ref_list: deref_text = text

    return deref_text, ref_list

def tokenizeText(text):
    doc = nlp(text)
    sentences = [sent.string.strip() for sent in doc.sents]
    sent_tokens = []
    tokens = []
    for sent in sentences:
        sent_doc = nlp(sent)
        token_one_sent = [token.string.strip() for token in sent_doc]
        sent_tokens.append(token_one_sent)
        tokens += token_one_sent
    return sent_tokens, tokens

# return the difference indices starting at 0.
def diffRevision(parent_sent_list, sent_list, context_size = 5):

    # make diff
    origin_start_idx, origin_start_idx = -1, -1
    target_start_idx, target_end_idx = -1, -1
    origin_diff_list, target_diff_list = [], []
    diffoutput = difflib.context_diff(parent_sent_list, sent_list, 'origin', 'target', n=context_size)
    for line in diffoutput:
        #print(line)
        # parse the origin diff line range: e.g., --- 56,62 ----
        if line.startswith("*** ") and line.endswith(" ****\n"):
            target_start_idx, target_end_idx = -1, -1 # reset the target indices
            range_line = line[4:-6]
            if ',' not in range_line:
                origin_start_idx = int(range_line)
                origin_end_idx = origin_start_idx
            else:
                origin_start_idx, origin_end_idx = [int(i) for i in range_line.split(',')]
            origin_sent_idx = origin_start_idx
            continue

        # parse the diff line range: e.g., --- 56,62 ----
        if line.startswith("--- ") and line.endswith(" ----\n"):
            origin_start_idx, origin_end_idx = -1, -1 # reset the origin indices
            range_line = line[4:-6]
            if ',' not in range_line:
                target_start_idx = int(range_line)
                target_end_idx = target_start_idx
            else:
                target_start_idx, target_end_idx = [int(i) for i in range_line.split(',')]
            target_sent_idx = target_start_idx
            continue

        if origin_start_idx >= 0:
            if len(line.strip('\n')) == 0:
                continue
            elif line.startswith('-') or line.startswith('!'):
                origin_diff_list.append(origin_sent_idx - 1) # adding the index starting at 0
                origin_sent_idx += 1
            else:
                origin_sent_idx += 1

        if target_start_idx >= 0:
            if len(line.strip('\n')) == 0:
                continue
            elif line.startswith('+') or line.startswith('!'):
                target_diff_list.append(target_sent_idx - 1) # adding the index starting at 0
                target_sent_idx += 1
            else:
                target_sent_idx += 1

    #print("Extracted Diff:", diff_sent_list)
    return origin_diff_list, target_diff_list

def findSentDiff(sents, tokens, token_diff):
    diff_idx, sent_idx, token_offset = 0, 0, 0
    diff_sents = set()
    if len(token_diff) == 0 or len(sents) == 0:
        return list(diff_sents)

    token_offset = len(sents[0])
    while diff_idx < len(token_diff) and sent_idx < len(sents):
        if token_offset >= token_diff[diff_idx]:
            cur_token = tokens[diff_idx]
            # avoid the case that one more sentence added because only one markup included.
            if len(cur_token) > 1:
                diff_sents.add(sent_idx)
            diff_idx += 1
        else:
            sent_idx += 1
            token_offset += len(sents[sent_idx])
        
    return list(diff_sents)


def extContext(tokens, token_diff, ctx_window):
    '''
    Extend the context into the token difference

    :param token_diff: a list of tokens which represents the difference between previous version and current version.
           ctx_window: the size of context before and after the edits
    :return:

    For example: token_diff = [2, 3, 4, 11, 12, 13, 14, 16] and ctx_window = 2
    The function will return ctx_tokens = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
                             action =     [0, 0, 1, 1, 1, 0, 0, 0,  0,  1,  1,  1,  1,  0,  1,  0,  0]
    '''


    ctx_set = set(token_diff)

    for idx in token_diff:
        for i in range(idx - ctx_window, idx + ctx_window + 1):
            if i < 0 or i >= len(tokens):
                continue
            ctx_set.add(i)

    action = []
    ctx_token = []
    ctx_token_idx = sorted(list(ctx_set))
    diff_set = set(token_diff)

    for i in ctx_token_idx:
        ctx_token.append(tokens[i])
        if i in diff_set:
            action.append(1)
        else:
            action.append(0)

    return ctx_token, action



# def findDiffSents(sents, diff_list):
#     token_idx = 0
#     start_sent_idx, end_sent_idx = -1, -1
#     for i, sent in enumerate(sents):
#         token_idx += len(sent)
#         if start_sent_idx < 0 and diff_list[0] < token_idx:
#             start_sent_idx = i
#         if end_sent_idx < 0 and diff_list[-1] < token_idx:
#             end_sent_idx = i
#         if start_sent_idx >= 0 and end_sent_idx >= 0:
#             break
#     return start_sent_idx, end_sent_idx

def extDiffSents(sents, start, end, max_tokens=200, max_sents_add=2):
    token_size = 0
    for i in range(start, end + 1):
        token_size += len(sents[i])

    context_sents = sents[start:end + 1]
    sents_head_added, sents_tail_added = 0, 0
    while token_size <= max_tokens and len(context_sents) < len(sents)\
        and sents_head_added < max_sents_add and sents_tail_added < max_sents_add:
        if start > 0:
            start -= 1
            insert_sent = sents[start]
            context_sents.insert(0, insert_sent)
            token_size += len(insert_sent)
            sents_head_added += 1

        if end < len(sents) - 1:
            end += 1
            insert_sent = sents[end]
            context_sents.append(insert_sent)
            token_size += len(insert_sent)
            sents_tail_added += 1

    diff_offset = sum([len(sents[i]) for i in range(start)])
    return context_sents, diff_offset

def mapDiffContext(sents, start_sent, end_sent):
    
    start_idx = -1
    end_idx = -1
    start_sent_str = " ".join(start_sent)
    end_sent_str = " ".join(end_sent)
    for i, sent in enumerate(sents):
        sent_str = " ".join(sent)
        if start_idx < 0 and start_sent_str in sent_str:
            start_idx = i
        
        if end_sent_str in sent_str:
            end_idx = i
            break
    
    # if start_idx == -1:
    #     start_idx = 0

    # if end_idx == -1:
    #     context_sents = sents[start_idx:]
    # else:
    #     context_sents = sents[start_idx:end_idx + 1]
    if start_idx == -1 or end_idx == -1:
        return None, None

    diff_offset = sum([len(sents[i]) for i in range(start_idx)])
    return sents[start_idx:end_idx + 1], diff_offset


# def extDiffContextInt(target_context_sents, target_sents, target_start, target_end, token_size, max_tokens=200, max_sents_add=2):
#     sents_head_added, sents_tail_added = 0, 0
    
#     return target_context_sents

def calTokenSize(sents):
    return sum([len(sent) for sent in sents])

def isSameSent(origin_sent, target_sent):

    origin_size = len(origin_sent)
    target_size = len(target_sent)

    if origin_size != target_size:
        return False

    for i in range(origin_size):
        if origin_sent[i] != target_sent[i]:
            return False
    return True

def stripContext(origin_sents, target_sents, max_token_size=200):

    start_match = True
    end_match = True
    origin_token_size = calTokenSize(origin_sents)
    target_token_size = calTokenSize(target_sents)
    diff_offset = 0
    while origin_token_size > max_token_size or target_token_size > max_token_size:

        if len(origin_sents) == 0 or len(target_sents) == 0:
                break

        if start_match and isSameSent(origin_sents[0], target_sents[0]):
            # remove the sentence from both origin and target
            sent_size = len(origin_sents[0])
            origin_sents = origin_sents[1:]
            target_sents = target_sents[1:]
            diff_offset += sent_size
            origin_token_size -= sent_size
            target_token_size -= sent_size
        else:
            start_match = False

        if len(origin_sents) == 0 or len(target_sents) == 0:
            break

        if end_match and isSameSent(origin_sents[-1], target_sents[-1]):
            sent_size = len(origin_sents[-1])
            origin_sents = origin_sents[:-1]
            target_sents = target_sents[:-1]
            origin_token_size -= sent_size
            target_token_size -= sent_size
        else:
            end_match = False

        if not start_match and not end_match:
            break

    if origin_token_size > max_token_size or target_token_size > max_token_size:
        origin_sents, target_sents = None, None

    return origin_sents, target_sents, diff_offset


def extractDiffContext(origin_sents, target_sents, origin_diff, target_diff, max_tokens=200):

    origin_context, target_context, diff_offset = stripContext(origin_sents, target_sents)
    if origin_context != None and target_context != None:
        origin_diff = [i - diff_offset for i in origin_diff]
        target_diff = [i - diff_offset for i in target_diff]

        # fix the issue in the case that the appended dot belonging to current sentence is marked as the previous sentence. See example:
        # + .
        # + Warren
        # + ,
        # ...
        # + -
        # + syndicalists
        #   .
        if len(target_diff) > 0 and target_diff[0] == -1:
            target_diff = target_diff[1:]
    
    return origin_context, target_context, origin_diff, target_diff


def filterRevision(comment, diff_list, max_sent_length):
    filter_pattern = "^.*\s*\W*(revert|undo|undid)(.*)$"
    filter_regex = re.compile(filter_pattern, re.IGNORECASE)
    
    if len(diff_list) == 0 or len(diff_list) > max_sent_length:
        return True
    comment = comment.strip()
    if not comment.startswith('/*'):
        return True
    elif comment.startswith('/*') and comment.endswith('*/'):
        return True
    filter_match = filter_regex.match(comment)
    if filter_match:
        return True
    
    return False
    #return filter_match

comment_pattern = "^/\*(.+)\*/(.*)$"
comment_regex = re.compile(comment_pattern)
def extractSectionTitle(comment):
    sect_title, sect_cmnt = '', comment
    comment_match = comment_regex.match(comment)
    if comment_match:
        sect_title = comment_match.group(1).strip()
        sect_cmnt = html.unescape(comment_match.group(2).strip()).strip()
    return sect_title, sect_cmnt

# TODO: escape sect_title at some point in case section titles contain markup
# e.g. /* Sejm of the [[Kingdom of Polan]] and the [[Polish-Lithuanian Commonwealth]] */
def extractSectionText(text, sect_title):
    sect_content = ''
    text_match = re.search('(=+)\s*' + sect_title + '\s*(=+)', text)

    if text_match:
        sect_sign = text_match.group(1)
        sect_sign_end = text_match.group(2)

        if sect_sign != sect_sign_end:
            print("ALERT: Section Data Corrupted!! Skip!!")
            return sect_content
            
        sect_start = text_match.regs[2][1]
        remain_sect = text[sect_start:].strip().strip('\n')

        # TODO: Fix the bug of comparing the ===Section Title=== after ==Section==
        next_sect_match = re.search(sect_sign + '.*' + sect_sign, remain_sect)
        if next_sect_match:
            sect_end = next_sect_match.regs[0][0]
            sect_content = remain_sect[:sect_end].strip().strip('\n')
        else:
            sect_content = remain_sect

    return sect_content


'''
Merge the sample outputs of all the dump files
'''
def mergeOutputs(output_path):

    # merge sample results
    print("Merging the sampled outputs from each files ...")
    sample_list = glob.glob(output_path + '*.json')
    sample_file = open(output_path + 'wikicmnt.json', "w", encoding='utf-8')
    for fn in tqdm(sample_list):
        with open(fn, 'r', encoding='utf-8') as fi:
            sample_file.write(fi.read())
