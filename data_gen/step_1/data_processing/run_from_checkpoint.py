"""
Runs a partial pipeline from a set of NDJson checkpoint files
"""

import os, io, argparse
import logging
import urllib
import json

from functools import partial
from tqdm import tqdm
from generator_chaining import process
from profiling import Profiled
from custom_filters import *
from custom_extractors import *
from generic_extractor import *

def scriptdir(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

def ndjson_generator(input_stream):
    for line in input_stream:
        object = json.loads(line)
        yield object

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-path', type=str, default="./data/in/", help='the input directory')
    parser.add_argument('--output-path', type=str, default="./data/out/", help='the output directory')
    parser.add_argument('--index', type=int, help='the index of the file in the input path to process')
    parser.add_argument('--prejoined-cc-index', type=str, default=None, help='optional file containing a pre-joined CommonCrawl index in CDX format')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG, format='(%(threadName)s) %(message)s')
    checkpoint_file = os.path.join(args.input_path, "{}.json".format(args.index))
    output_file = os.path.join(args.output_path, "{}.json".format(args.index))
    checkpoint_input_stream = open(checkpoint_file, "rt", encoding='utf-8')
    json_output_stream = open(output_file, "w", buffering=1, encoding='utf-8')

    ### chose processing and filtering steps here:
    processors = [
        ## Start processing from here ##
        extract_common_crawl_groundings(prejoined_index_file=args.prejoined_cc_index),
        remove_without_grounding_docs,
        clean_newlines(fields=["grounding_docs"]),
        extract_grounding_snippet(target_length=200, min_overlap_tokens=5),
        filter_list_by_language(languages = ['en'], field="grounding_snippets"), # only keep english. can also be run on grounding_docs instead
        remove_without_grounding_snippets,
        project_to_fields([
            'rev_id', 'page_id', 'parent_id', 'timestamp',
            'src_text', 'tgt_text', 'comment_text',
            'section_title', 'page_title',
            'diff_url',
            'src_tokens', 'tgt_tokens', 'src_action', 'tgt_action',
            'left_context', 'right_context', "left_text", "right_text",
            'grounding_urls', "grounding_canonical_urls",
            "grounding_docs", # this is huge..
            "grounding_snippets"]),
        save_to_disk(json_output_stream, NDJsonExtractor()), # chose extractor here
    ]
    
    process(
        checkpoint_input_stream,
        base_generator = ndjson_generator,
        processors=processors
    )
    
    checkpoint_input_stream.close()
    json_output_stream.close()
    logging.info("Done with task %d" % args.index)
    logging.info(Profiled.summarize(processors))