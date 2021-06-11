# TODO: Fix references to include self closing tags (<ref name=... />)

from dotenv import load_dotenv
load_dotenv()

import sys
import datetime
import bz2
import logging
import queue
import json
import argparse
import os
import io
import html
import copy
import itertools
from nltk.tokenize import word_tokenize, sent_tokenize
from sentence_splitter import SentenceSplitter
from tempfile import TemporaryFile, NamedTemporaryFile
from profanityfilter import ProfanityFilter

from wiki_util import *
#from wiki_dump_download import existFile
#from run_all_processing import process, scriptdir
from custom_extractors import NDJsonExtractor
from generic_extractor import generate_revision_pairs
from generator_chaining import chain_generators
from wikitext_processing import clean_wiki_text
from custom_filters import exclude_page_types, has_section_title, has_comment

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

def unescapeHTML(revision):
    revision['src_text'] = html.unescape(revision['src_text'])
    revision['tgt_text'] = html.unescape(revision['tgt_text'])
    yield revision

def retrieveSection(revision):
#    def _retrieveSection(text, title):
#        sections, titles = split_into_sections(text)
#        try:
#            sect_idx = titles.index(title)
#        except ValueError:
#            return ''
#        return sections[sect_idx]

    title = re.escape(revision['section_title']) #important to escape title
    src_sect = extractSectionText(revision['src_text'], title)
    tgt_sect = extractSectionText(revision['tgt_text'], title)

    #only yield if found section in both source and target texts
    if src_sect and tgt_sect: 
        # remove some unused fields, restructure revision
        revision.pop('src_text')
        revision.pop('tgt_text')
        revision['source'] = {'text': src_sect}
        revision['target'] = {'text': tgt_sect}

        yield revision

def splitSections(revision):
    source_sections, source_titles = split_into_sections(revision['src_text'])
    target_sections, target_titles = split_into_sections(revision['tgt_text'])
    
    tgt_sect_dict = {title: text for title,text in zip(target_titles, target_sections)}

    for src_sect, src_title in zip(source_sections, source_titles):
        # recover matching target sections
        try: tgt_sect = tgt_sect_dict[src_title]
        except KeyError: continue

        sect_rev = {
            'rev_id': revision['rev_id'],
            'parent_id': revision['parent_id'],
            'page_id': revision['page_id'],
            'page_title': revision['page_title'],
            'comment_text': revision['comment_text'],
            'section_title': src_title,
            'source': {
                'text': src_sect
                },
            'target': {
                'text': tgt_sect
                }
            }

        yield sect_rev

def processRefs(revision): #TODO: extract urls from refs and merge source and target refs
    revision['source']['text'], _ = retrieveReferences(revision['source']['text'])
    revision['target']['text'], revision['refs'] = retrieveReferences(revision['target']['text'])
    # right now use target refs
    yield revision

def stripMarkup(revision):
#    revision['source']['text'] = str(mwparserfromhell.parse(revision['source']['text']).strip_code())
#    revision['target']['text'] = str(mwparserfromhell.parse(revision['target']['text']).strip_code())
    revision['source']['text'] = ' '.join(clean_wiki_text(revision['source']['text']))
    revision['target']['text'] = ' '.join(clean_wiki_text(revision['target']['text']))

    yield revision

def tokenize(revision):
    splitter = SentenceSplitter(language='en') # might want to pass this object by reference?

    #def _tokenize(text):
    #    return [word_tokenize(s) for s in splitter.split(text) if len(s)]

#    revision['source']['sentences'] = [s for s in sent_tokenize(revision['source']['text'])\
#            if len(s)]
#    revision['target']['sentences'] = [s for s in sent_tokenize(revision['target']['text'])\
#            if len(s)]
    revision['source']['sentences'] = [s for s in splitter.split(revision['source']['text'])\
            if len(s)]
    revision['target']['sentences'] = [s for s in splitter.split(revision['target']['text'])\
            if len(s)]
   
    revision['source']['tokens'] = [word_tokenize(s) for s in revision['source']['sentences']]
    revision['target']['tokens'] = [word_tokenize(s) for s in revision['target']['sentences']]
    yield revision

def matchSentences(k_s=5, k_c=5, cutoff=0.1):
    def generator(revision):

        if abs(len(revision['source']['tokens']) - len(revision['target']['tokens'])) > k_s:
            raise(StopIteration)

        def match_sequences(a, b):
            matches = {}
            for i, s in enumerate(a):
                matches[i] = []
                
                min_idx = max(i-k_s, 0)
                max_idx = min(i+k_s, len(b))

                if min_idx >= max_idx: continue
                
                for j in range(min_idx, max_idx):
                    if sentence_bleu([s], b[j], (0.25,0.25,0.25,0.25)) > cutoff:
                        matches[i].append(j)
            return matches

        target_matches = match_sequences(revision['target']['tokens'], revision['source']['tokens'])
        source_matches = match_sequences(revision['source']['tokens'], revision['target']['tokens'])

        edges = {}
        for i in source_matches:
            edges[i] = [j + len(source_matches) for j in source_matches[i]]
        for j in target_matches:
            edges[j+len(source_matches)] = target_matches[j]

        # breadth-first search to find connected components
        components = {}
        colors = {i: 0 for i in range(len(source_matches) + len(target_matches))}
        def DFS(root, component_id):
            assert component_id not in components, "component_id already exists in components"
            nodes_to_explore = queue.Queue()
            nodes_to_explore.put(root)
            colors[root] = 1
            components[component_id] = [root]
            while not nodes_to_explore.empty():
                v = nodes_to_explore.get()
                for u in edges[v]:
                    if colors[u] != 1:
                        nodes_to_explore.put(u)
                        colors[u] = 1
                        components[component_id].append(u)
        component_id = 0
        for node in range(len(source_matches) + len(target_matches)):
            if colors[node] != 1:
                DFS(node, component_id)
                component_id += 1
        
        # for each source (resp. target) sentence, keep track of the previous (resp. next sentence 
        # which isn't a deletion (resp. insertion)
        previous_conserved_source_sentence = {}
        for s in range(len(source_matches)):
            if source_matches[s]:
                previous_conserved_source_sentence[s] = s
            else:
                if s > 0:
                    previous_conserved_source_sentence[s] = previous_conserved_source_sentence[s-1]
                else:
                    previous_conserved_source_sentence[s] = -1

        next_conserved_target_sentence = {}
        for t in range(len(target_matches))[::-1]:
            if target_matches[t]:
                next_conserved_target_sentence[t] = t
            else:
                if t < (len(target_matches)-1):
                    next_conserved_target_sentence[t] = next_conserved_target_sentence[t+1]
                else:
                    next_conserved_target_sentence[t] = -1
        
        for c in components:
            source_idxs = []
            target_idxs = []
            for v in components[c]:
                if v >= len(source_matches):
                    target_idxs.append(v - len(source_matches))
                else:
                    source_idxs.append(v)

            source_sent = ''
            source_tokens = []
            target_sent = ''
            target_tokens = []

            if len(source_idxs) > 1 or len(target_idxs) > 1: #don't know how to treat this case
                source_right_context = -1
                target_left_context = -1
            elif not source_idxs:
                ncts = next_conserved_target_sentence[target_idxs[0]]
                if ncts < 0:
                    source_right_context = -1
                else:
                    source_right_context = target_matches[ncts][0]

                target_left_context = target_idxs[0] - 1

                target_sent = revision['target']['sentences'][target_idxs[0]]
                target_tokens = revision['target']['tokens'][target_idxs[0]]
            elif not target_idxs:
                source_right_context = source_idxs[0] + 1 if source_idxs[0] < len(source_matches)\
                        else -1
                pcss = previous_conserved_source_sentence[source_idxs[0]]
                if pcss < 0:
                    target_left_context = -1
                else:
                    target_left_context = source_matches[pcss][0]

                source_sent = revision['source']['sentences'][source_idxs[0]]
                source_tokens = revision['source']['tokens'][source_idxs[0]]
            else:
                source_right_context = source_idxs[0] + 1 if source_idxs[0] < len(source_matches)\
                        else -1
                target_left_context = target_idxs[0] - 1

                source_sent = revision['source']['sentences'][source_idxs[0]]
                source_tokens = revision['source']['tokens'][source_idxs[0]]
                target_sent = revision['target']['sentences'][target_idxs[0]]
                target_tokens = revision['target']['tokens'][target_idxs[0]]

            if source_right_context >= 0:
                right_context = list(range(source_right_context, min(len(source_matches), source_right_context + k_c)))
            else:
                right_context = []

            if target_left_context >= 0:
                left_context = list(range(max(0, target_left_context - k_c + 1), target_left_context + 1))
            else:
                left_context = []
            
            right_context_sentences = []
            for i in right_context:
                right_context_sentences.append(revision['source']['sentences'][i])
            left_context_sentences = []
            for i in left_context:
                left_context_sentences.append(revision['target']['sentences'][i])

            #components[c] = (source_idxs, target_idxs, right_context, left_context)
            sent_revision = copy.deepcopy(revision) # deep copy revision
            sent_revision['source'] = {
                    'sentence': source_sent,
                    'tokens': source_tokens,
                    'text': revision['source']['text']}
            sent_revision['target'] = {
                'sentence': target_sent,
                'tokens': target_tokens,
                'text': revision['target']['text']}
            sent_revision['left_context'] = left_context_sentences
            sent_revision['right_context'] = right_context_sentences

            yield sent_revision


        return components

        
#        target_matches = {}
#        for i, s in enumerate(revision['source']['tokens']):
#            # and a dummy empty sentence with score=cutoff for deletions
#            matches = ['']
#            bleu_scores = [cutoff]
#
#            min_idx = max(i-k_s, 0)
#            max_idx = min(i+k_s, len(revision['target']['tokens']))
#
#            if min_idx >= max_idx: continue
#            
#            for j in range(min_idx, max_idx):
#                bleu_scores.append(sentence_bleu([s], revision['target']['tokens'][j], (0.25,0.25,0.25,0.25)))
#                matches.append(

#        for i in range(len(revision['source']['tokens']) - 1):
#            for j in range(i+1, min(i + 3, len(revision['source']['tokens']))):
#
#                def retrieve_context(revision, src_left, src_right, tgt_left, tgt_right):
#
#                    left_src_ctx = revision['source']['sentences'][max(src_left-k_c, 0):src_left]
#                    left_tgt_ctx = revision['target']['sentences'][max(tgt_left-k_c,0):tgt_left]
#
#                    right_src_ctx = revision['source']['sentences'][src_right:min(len(revision['source']['sentences']), src_right+k_c)]
#                    right_tgt_ctx = revision['target']['sentences'][tgt_right:min(len(revision['target']['sentences']), tgt_right+k_c)]
#
#                    return left_src_ctx, left_tgt_ctx, right_src_ctx, right_tgt_ctx
#
#                for k in sentence_map[i]:
#                    for l in sentence_map[j]:
#
#                        if (l - k) == 1 and (j-i) == 2: # sentence deletion
#                            source_sent = revision['source']['sentences'][j-1] 
#                            source_tokens = revision['source']['tokens'][j-1]
#                            target_sent = ''
#                            target_tokens = []
#                            left_src_ctx, left_tgt_ctx, right_src_ctx, right_tgt_ctx = retrieve_context(revision,
#                                   j-1, l, j, l)
#                        elif (l - k) == 2 and (j-i) == 1: # sentence insertion
#                            source_sent = ''
#                            source_tokens = []
#                            target_sent = revision['target']['sentences'][l-1]
#                            target_tokens = revision['target']['tokens'][l-1]
#                            left_src_ctx, left_tgt_ctx, right_src_ctx, right_tgt_ctx = retrieve_context(revision,
#                                    j, l - 1, j, l)
#                        elif (l - k) == 2 and (j-i) == 2: # sentence substitution
#                            # check the the sentence is edited
#                            if revision['source']['sentences'][j-1] == revision['target']['sentences'][l-1]: continue
#
#                            source_sent = revision['source']['sentences'][j-1]
#                            source_tokens = revision['source']['tokens'][j-1]
#                            target_sent = revision['target']['sentences'][l-1]
#                            target_tokens = revision['target']['tokens'][l-1]
#                            left_src_ctx, left_tgt_ctx, right_src_ctx, right_tgt_ctx = retrieve_context(revision,
#                                    j-1, l-1, j, l)
#                        else: continue

#                        sent_revision = copy.deepcopy(revision) # deep copy revision
#                        sent_revision['source'] = {
#                                'sentence': source_sent,
#                                'tokens': source_tokens,
#                                'left_context': left_src_ctx,
#                                'right_context': right_src_ctx,
#                                'sentences': revision['source']['sentences']}
#                        sent_revision['target'] = {
#                            'sentence': target_sent,
#                            'tokens': target_tokens,
#                            'left_context': left_tgt_ctx,
#                            'right_context': right_tgt_ctx,
#                            'sentences': revision['target']['sentences']}
#
#                        yield sent_revision

    return generator



def splitSentences(k=5,k_c=5):
    def generator(revision):
        for i, s in enumerate(revision['source']['tokens']):
            min_idx = max(i-k, 0)
            max_idx = min(i+k, len(revision['target']['tokens']))

            if min_idx >= max_idx: continue

            bleu_scores = [sentence_bleu([s], revision['target']['tokens'][j])\
                    for j in range(min_idx, max_idx)]

            match_idx = np.argmax(bleu_scores) + min_idx
            match_score = np.max(bleu_scores)
            src_sent = revision['source']['sentences'][i]
            tgt_sent = revision['target']['sentences'][match_idx]
            tgt_tokens = revision['target']['tokens'][match_idx]
            
            left_src_ctx = revision['source']['sentences'][max(i-k_c, 0):i]
            left_tgt_ctx = revision['target']['sentences'][max(match_idx-k_c,0):match_idx]

            right_src_ctx = revision['source']['sentences'][i:min(len(revision['source']['sentences']), i+k_c)]
            right_tgt_ctx = revision['target']['sentences'][match_idx:min(len(revision['target']['sentences']), match_idx+k_c)]
            
            # deep copy is a potential slow down. Can speed up at the cose
            # of making the code less maintainable (i.e. select which fields
            # to copy right here...)
            sent_revision = copy.deepcopy(revision) # deep copy revision
            sent_revision['source'] = {
                    'sentence': src_sent,
                    'tokens': s,
                    'left_context': left_src_ctx,
                    'right_context': right_src_ctx}
            sent_revision['target'] = {
                'sentence': tgt_sent,
                'tokens': tgt_tokens,
                'left_context': left_tgt_ctx,
                'right_context': right_tgt_ctx,
                'match_score': match_score}

            yield sent_revision

    return generator

def profanityFilter():
    pf = ProfanityFilter()
    def generator(revision):
        has_profanity = pf.has_bad_word(revision['source']['sentence']) or\
                pf.has_bad_word(revision['target']['sentence'])
        if not has_profanity:
            yield revision

    return generator

def diffText(revision):
    revision['source']['diff'], revision['target']['diff'] = diffRevision(
            revision['source']['tokens'], revision['target']['tokens'])
    yield revision

def filterForEdits(revision):
    if revision['source']['sentence'] != revision['target']['sentence']:
        yield revision

def filterPunctuation(blacklist = ['[',']','|','/','\\','$','#','@','%']):
    def generator(revision):
        yield_revision = True
        for c in blacklist:
            if c in revision['source']['sentence'] or c in revision['target']['sentence']:
                yield_revision = False
                break
        if yield_revision: yield revision

    return generator 

def filterContiguousEdits(revision):
    if revision['source']['diff'] and not revision['target']['diff'] \
            and is_contiguous(revision['source']['diff']):
                revision['type'] = 'deletion'
                yield revision
    elif not revision['source']['diff'] and revision['target']['diff'] \
            and is_contiguous(revision['target']['diff']):
                revision['type'] = 'insertion'
                yield revision

def process(input_stream, output_stream, extractor, base_generator, processors, max_edits=0):
    iterable = tqdm(base_generator(input_stream), "baseline generator", mininterval=3.0)
    results = chain_generators(iterable, processors)
    if max_edits > 0: # 0 is special value to process everything
        results = itertools.islice(results, 0, max_edits)
    for instance in tqdm(results, "final results", mininterval=3.0):
        extractor.write_instance(output_stream, instance)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, help='index of file to process', default=1)
    parser.add_argument('--input_dir', type=str, help='directory on disk to read from',
            default='/mnt/nlp-storage/data/raw/wikipedia-subsample/')
    parser.add_argument('--output_dir', type=str, help='directory on disk to write to',
            default='/mnt/nlp-storage/data/processed/atomic-edits/temp/')
    parser.add_argument('--azure', action='store_true', default=False, help='whether to read from azure')
    parser.add_argument('--container', type=str, help='aws container', default='wikipedia-data')
    parser.add_argument('--blob_path', type=str, help='blob prefix on azure',
            default='raw/wikipedia-subsample')
    parser.add_argument('--out_path', type=str, default='processed/atomic-edits/latest/')
    parser.add_argument('--max_edits', type=int, default=0)
    parser.add_argument('--temp_dir', type=str, default='tmp') # TODO: remove this line?
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG, format='(%(threadName)s) (%(name)s) %(message)s')
    logger = logging.getLogger('azure')
    logger.setLevel(logging.ERROR)
    logger = logging.getLogger('urllib3')
    logger.setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)

    
    if args.azure:
        # set up azure connection
        connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        
        container_client = blob_service_client.get_container_client(args.container)
        in_blobs = [b.name for b in container_client.list_blobs() if args.blob_path in b.name]
        dump_blob = in_blobs[args.index]

        logger.debug('processing file: {}'.format(dump_blob))

        input_file = NamedTemporaryFile()
        output_file = NamedTemporaryFile()
        
        blob_client = blob_service_client.get_blob_client(container=args.container,\
                blob=dump_blob)
        
        with open(input_file.name, 'wb') as f:
            f.write(blob_client.download_blob().readall())
        
        wiki_input_stream = bz2.open(input_file.name, 'rt', encoding='utf-8')
        json_output_stream = open(output_file.name, 'w', buffering=1, encoding='utf-8')

    else:
        in_files = glob.glob(os.path.join(args.input_dir, "*.txt.bz2"))
        in_file = in_files[args.index]

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        out_file = os.path.join(args.output_dir, 'enwiki-sample-' + os.path.basename(in_file)[27:-8] + '.json')
        wiki_input_stream = bz2.open(in_file, 'rt', encoding='utf-8')
        json_output_stream = open(out_file, 'w', buffering=1, encoding='utf-8')
    


    process(
        wiki_input_stream,
        json_output_stream,
        extractor = NDJsonExtractor(),
        base_generator = generate_revision_pairs,
        processors = [
            exclude_page_types(["Talk:", "User", "Wikipedia"]),
            has_section_title,
            has_comment,
            unescapeHTML,
            retrieveSection,
   #         splitSections,
            processRefs,
            stripMarkup,
            tokenize,
            splitSentences(k=5, k_c=20),
            diffText,
            filterContiguousEdits
        ],
        max_edits = args.max_edits
    )

    wiki_input_stream.close()
    json_output_stream.close()
   
    if args.azure:
        out_blob = args.out_path + 'enwiki-sample-' + os.path.basename(dump_blob)[27:-8] + '.json'
        blob_client = blob_service_client.get_blob_client(container=args.container,\
                blob=out_blob)
        with open(output_file.name, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)

        input_file.close() # deletes temp file
        output_file.close()
 
