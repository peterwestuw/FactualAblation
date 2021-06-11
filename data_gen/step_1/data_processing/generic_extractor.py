#!/usr/bin/env python

import sys
import datetime
import logging
import json
import os
import io
import re
import itertools
import mwparserfromhell
from itertools import accumulate
from operator import add
from copy import deepcopy
from nltk.tokenize import word_tokenize, sent_tokenize

from wiki_util import *
from profiling import Profiled

def generate_revision_pairs(wiki_stream, max_bytes=None):
    logger=logging.getLogger(__name__)
    start_time = datetime.datetime.now()

    revision_count = 0
    page_count = 0
    prev_text = '' # changed from None!
    prev_page_title = ''

    records = split_records(wiki_stream, max_bytes=max_bytes)

    for page_title, page_id, revision in records:
        revision_count += 1

        # fields 
        rev_id, parent_id, timestamp, username, userid, userip, comment, text = extract_data(revision)
        comment = cleanCmntText(comment)
        sect_title, comment = extractSectionTitle(comment)
        if prev_page_title != page_title:
            page_count += 1
            prev_page_title = page_title
            prev_text = text

        else:
            meta = {
                "rev_id": rev_id,
                "page_id": page_id,
                "parent_id": parent_id,
                "src_text": prev_text,
                "tgt_text": text,
                "comment_text":comment,
                "section_title":sect_title,
                "page_title": page_title,
                "timestamp": timestamp
            }

            meta['diff_url'] = 'https://en.wikipedia.org/w/index.php?title=' + \
                meta["page_title"].replace(" ",'%20') + '&type=revision&diff=' + meta["rev_id"] + '&oldid=' + meta["parent_id"]

            # for next iteration, current text becomes prev_text
            prev_text = text

            yield meta

    time_elapsed = datetime.datetime.now() - start_time
    logger.debug("=== iterated through " + str(revision_count) + " revisions " \
                    + 'across ' + str(page_count) + ' pages. ' \
                    + 'Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed) + ' ===')

@Profiled.generator
def restrict_to_section(meta):
    """Cuts the text (source, target) to only contain the section mentioned in the comment"""
    try:
        if not meta["section_title"]:
            return

        meta["src_text"] = extractSectionText(meta["src_text"], meta["section_title"])
        meta["tgt_text"] = extractSectionText(meta["tgt_text"], meta["section_title"])

        # only yield pairs that are not empty
        if meta["src_text"] and meta["tgt_text"]:
            yield meta

    except:
        # don't yield anything if any exception happens
        logging.error("Regex error in " + str(meta['page_id']))
        return


@Profiled.generator
def clean_markup_mediawikiparser(instance):
    def remove_refs(s): return re.sub(r"</?ref[^>]*>", "", s)
    def remove_misc(s): return re.sub(r"``|''", "", s)
    def parse(s): return remove_misc(remove_refs(str(mwparserfromhell.parse(s).strip_code())))

    try:
        instance['src_text'] = parse(instance['src_text'])
        instance['tgt_text'] = parse(instance['tgt_text'])
        yield instance
    except Exception as e:
        logging.error("Could not run mwparserfromhell: " + str(e))
        return

@Profiled.generator
def clean_markup_custom(instance):
    instance['src_text'] = cleanWikiText(instance['src_text'])
    instance['tgt_text'] = cleanWikiText(instance['tgt_text'])
    yield instance

def tokenize(mode):
    def nltk_version(text):
        sentences = [word_tokenize(s) for s in sent_tokenize(text) if len(s)]
        flattened = [t for s in sentences for t in s]
        return sentences, flattened

    tokenize_impl = {"nltk": nltk_version, "spacy": tokenizeText}[mode]
    
    @Profiled.generator
    def tokenize(instance):
        instance['src_sents'], instance['src_tokens'] = tokenize_impl(instance['src_text'])
        instance['tgt_sents'], instance['tgt_tokens'] = tokenize_impl(instance['tgt_text'])
        if instance['src_sents'] == None or instance['tgt_sents'] == None:
            return
        yield instance
    
    return tokenize

@Profiled.generator
def prune_to_sentence_diff(instance):
    # create single-string sentence as a first step
    src_sents = [" ".join(x) for x in instance['src_sents']]
    tgt_sents = [" ".join(x) for x in instance['tgt_sents']]
    src_sent_diff, tgt_sent_diff = diffRevision(src_sents, tgt_sents)
    if len(src_sent_diff) == 0 and len(tgt_sent_diff) == 0:
        return

    if src_sent_diff == tgt_sent_diff:
        # prune context to only differing sentence
        instance['src_tokens'] = [token for diff_idx in src_sent_diff for token in instance['src_sents'][diff_idx]]
        instance['tgt_tokens'] = [token for diff_idx in tgt_sent_diff for token in instance['tgt_sents'][diff_idx]]

    yield instance

@Profiled.generator
def compute_diff(instance):
    """Given src_tokens and tgt_tokens, computes a token-level diff"""

    # extract the offset of the changed tokens in both src and tgt
    src_token_diff, tgt_token_diff = diffRevision(instance['src_tokens'], instance['tgt_tokens'])
    instance['tgt_token_diff'] = tgt_token_diff
    instance['src_token_diff'] = src_token_diff

    if len(src_token_diff) == 0 and len(tgt_token_diff) == 0:
        return

    if (len(src_token_diff) > 0 and src_token_diff[0] < 0) or (
            len(tgt_token_diff) > 0 and tgt_token_diff[0] < 0):
        return

    yield instance

def group_by_continuous(num_iterable):
    """Creates a nested iterable based on an iterable of numbers, where each sub-list is continuous"""
    buffer = []
    for i in num_iterable:
        if buffer and buffer[-1] + 1 == i:
            buffer.append(i)
        else:
            if buffer: yield buffer
            buffer = [i]
    if buffer: yield buffer

@Profiled.generator
def find_continous_edits(instance):
    """Helper processing step that identifies continuous edits"""
    instance['tgt_token_diffs'] = list(group_by_continuous(instance['tgt_token_diff']))
    instance['src_token_diffs'] = list(group_by_continuous(instance['src_token_diff']))
    yield instance

@Profiled.generator
def filter_single_edit_span(instance):
    """Filters down to edits that only have a single continuous edit token span"""
    if len(instance['tgt_token_diffs']) == 1:
        del instance['tgt_token_diffs']
        del instance['src_token_diffs']
        yield instance

@Profiled.generator
def split_into_continuous_edits(instance):
    """Forks an instance containining multiple edit locations within one edit into a seperate instance for each edit span"""
    edits = instance['tgt_token_diffs']
    if not edits: return
    #print("will split into {} subinstance".format(len(edits)))
    del instance['tgt_token_diffs']
    del instance['src_token_diffs']
    del instance['tgt_token_diff']
    instance['src_token_diff'] = []
    for edit_span in edits:
        sub_instance = deepcopy(instance)
        sub_instance['tgt_token_diff'] = edit_span
        yield sub_instance


def filter_additions(min_length, max_length):
    """Filters down to items where text has been ADDED, given a minimum and maximum token length"""

    @Profiled.generator
    def filter_additions(instance):
        len_tgt_diff = len(instance['tgt_token_diff'])
        len_src_diff = len(instance['src_token_diff'])
        len_src = len(instance['src_tokens'])
        len_tgt = len(instance['tgt_tokens'])
        tokens_added = len_tgt - len_src
        nothing_changed_in_src = len_src_diff == 0
        only_added = len_tgt_diff == tokens_added
        if nothing_changed_in_src and only_added and tokens_added >= min_length and tokens_added < max_length: 
            yield instance
    return filter_additions

def extract_context_around_diff(ctx_window_size):
    """Creates a context window around the diff, based on tokens to be added to the left/right"""
    
    @Profiled.generator
    def extract_context(instance):
        src_ctx_tokens, src_action = extContext(instance['src_tokens'], instance['src_token_diff'], ctx_window_size)
        tgt_ctx_tokens, tgt_action = extContext(instance['tgt_tokens'], instance['tgt_token_diff'], ctx_window_size)

        instance.update({"src_tokens": src_ctx_tokens, "src_action": src_action,
                        "tgt_tokens": tgt_ctx_tokens, "tgt_action": tgt_action})

        yield instance
    return extract_context


def extract_sentence_context_around_target(left_sentences=1, right_sentences=1):
    """Creates a context window around the TARGET tokens, based on sentence tokenization.
    By setting left_sentences and right_sentences to 0, this generator would only extract within-sentence left/right context.
    If a value > 0 is given, full sentences on the left/right are appended to the context"""

    def flatten1(lol):
        return [x for inner in lol for x in inner]

    def compute_token_offsets(sentences):
        lengths = [len(s) for s in sentences]
        offsets = [0] + list(accumulate(lengths))
        return offsets

    def find_sentence_indices(token_offsets, indices):
        def find_index(idx):
            for i, offset in enumerate(token_offsets):
                if offset > idx: return i-1
            return -1
        return [find_index(i) for i in indices]
    
    @Profiled.generator
    def extract_sentence_context_around_target(instance):
        token_diff = instance['tgt_token_diff']
        if not len(token_diff): return
        
        # find sentence index based on token diff indices
        min_token, max_token = token_diff[0], token_diff[-1]
        token_offsets = compute_token_offsets(instance['tgt_sents'])
        sent_indices = find_sentence_indices(token_offsets, [min_token, max_token])
        min_sentence, max_sentence = min(sent_indices), max(sent_indices)
        
        # extract left/right context sentence(s). Does not contain left/right context from target sentence
        instance["left_context"] = instance['tgt_sents'][min_sentence-left_sentences:min_sentence]
        instance["right_context"] = instance['tgt_sents'][max_sentence+1:max_sentence+1+right_sentences]

        # prune target to only contain relevant sentences
        start_token = token_offsets[min_sentence]
        end_token = token_offsets[max_sentence+1]
        
        # extract within-sentence left/right context
        in_sentence_left_context = instance["tgt_tokens"][start_token:min_token]
        in_sentence_right_context = instance["tgt_tokens"][max_token:end_token]
        in_sentence_target = instance["tgt_tokens"][min_token:max_token]

        instance["tgt_tokens"] = in_sentence_target
        instance["tgt_sents"] = instance['tgt_sents'][min_sentence:max_sentence+1]
        
        # hacky reconstruction of tgt_text
        instance["tgt_text"] = " ".join(instance["tgt_tokens"])
        instance["left_text"] = " ".join(flatten1(instance["left_context"]) + in_sentence_left_context) 
        instance["right_text"] = " ".join(in_sentence_right_context + flatten1(instance["right_context"]))

        # get rid of all src_ keys
        for key in [key for key in instance if key.startswith("src_")]: del instance[key]
        yield instance

    return extract_sentence_context_around_target


def filter_to_min_context(min_left_tokens = None, min_right_tokens = None):

    @Profiled.generator
    def filter_to_min_context(instance):
        if min_left_tokens:
            n_left_tokens = sum(map(len, instance["left_context"]))
            if n_left_tokens < min_left_tokens:
                return

        if min_right_tokens:
            n_right_tokens = sum(map(len, instance["right_context"]))
            if n_right_tokens < min_right_tokens:
                return

        yield instance

    return filter_to_min_context

def project_to_fields(accepted_fields):
    """Creates a projection of the instance to a pre-supplied set of fields"""
    accepted_fields = set(accepted_fields)

    @Profiled.generator
    def project_to_fields(instance):
        for key in list(instance):
            if not key in accepted_fields:
                del instance[key]
        yield instance
    return project_to_fields

