"""
Downloads (on-demand) a range of dump files and processes them according to filters and processors specified below.
"""

import os, io, argparse
import logging
import urllib

from functools import partial
from tqdm import tqdm
from azure_utils import open_azure_input_stream, upload_to_azure_output_stream
from wiki_dump_download import existFile, get_dump_task
from wikihelpers import estimate_article_count_from_filename

from profiling import Profiled
from generator_chaining import chain_generators
from custom_filters import *
from custom_extractors import *
from generic_extractor import *

def download_on_demand(url, dump_file, temp_path, compress_type):
    # Hack for HPC: cert verification issues
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    if not existFile(temp_path, dump_file, compress_type):
        urllib.request.urlretrieve(url, dump_file)
    else:
        logging.info("File Exists, Skip: " + url)
    return dump_file

def scriptdir(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

def process(input_stream, base_generator, processors):
    """Applies the base_generator on input_stream, then chains processor steps in processors, finally uses extractor to write to output_stream"""
    iterable = tqdm(base_generator(input_stream), "baseline generator", mininterval=3.0)
    results = chain_generators(iterable, processors)
    for _ in tqdm(results, "final results", mininterval=3.0):
        Profiled.total_count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int, help='the index of the file (within the dump status file) to download and process')
    parser.add_argument('--dumpstatus_path', type=str, default='./data/dumpstatus.json')
    parser.add_argument('--temp-path', type=str, default="./data/raw/", help='the temp / data directory, used to download wiki dump files')
    parser.add_argument('--output-path', type=str, default="./data/out/", help='the output directory')
    parser.add_argument('--compress-type', type=str, default='bz2', help='the compressed file type to download: 7z or bz2 [default: bz2]')
    parser.add_argument('--max_mb', type=int, default=None, help='if given, only processes the first max_mb MByte from the file and then stops')

    parser.add_argument('--azure', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG, format='(%(threadName)s) %(message)s')

    logging.info("Determining dump task..")
    dump_tasks = get_dump_task(args.dumpstatus_path, args.temp_path, args.compress_type, args.index, args.index, azure=False)
    url, dump_file, cur_progress, total_num = dump_tasks.assign_task()
    article_count = estimate_article_count_from_filename(dump_file)

    download_on_demand(url, dump_file, args.temp_path, args.compress_type)
    output_file = os.path.join(args.output_path, "{}.json".format(args.index))
    logging.info("Dumped to " + dump_file + " processing to " + output_file)
    if article_count: logging.info("Estimating count of {} articles".format(article_count))
    if args.max_mb: logging.info("Restricting to the first {} MByte of the raw XML".format(args.max_mb))

    wiki_input_stream = open_azure_input_stream() if args.azure else bz2.open(dump_file, "rt", encoding='utf-8')
    json_output_stream = io.StringIO() if args.azure else open(output_file, "w", buffering=1, encoding='utf-8')

    max_bytes = args.max_mb and 1024*1024* args.max_mb

    ### chose processing and filtering steps here:
    processors = [
        ## has_section_title,
        #comment_length(5, 200),
        exclude_page_types(["Talk:", "User talk:", "Wikipedia talk:", "Template talk:", "User:", "Wikipedia:"]),
        comment_blocklist_filter(["[[Project:AWB|AWB]]", "[[Project:AutoWikiBrowser|AWB]]", "Undid revision"]),
        #comment_token_length(2, 1000),
        text_length(5, 10000000),
        ## restrict_to_section,
        clean_urls(replacement='URL'),
        has_urls_in_text(look_in_src=True, look_in_tgt=True),
        ## grounding_domain_whitelist(file=scriptdir("domains-official.txt")), ## NOTE: disabled for now
        clean_markup_mediawikiparser,
        clean_markup_custom,
        clean_newlines(),
        tokenize(mode='nltk'), ## NOTE: mode can be 'spacy' or 'nltk'
        prune_to_sentence_diff,
        compute_diff,
        find_continous_edits,
        filter_single_edit_span, # alternative step: split_into_continuous_edits,
        filter_additions(min_length=3, max_length=100),
        restrict_grounding_to_max_distance(max_token_distance = 1), # only include urls in the edit
        has_grounding(), # abort here if there is no grounding documents left
        extract_sentence_context_around_target(1, 1), # original: extract_context_around_diff(ctx_window_size=5),
        filter_to_min_context(min_left_tokens=10), # require some left context, right context for now optional
        #canonize_grounding(), # convert grounding_urls into canonical grounding urls for CommonCrawl
        # extract_common_crawl_groundings(),
        # remove_without_grounding_docs,
        # extract_grounding_snippet(target_length=200, min_overlap_tokens=5),
        # filter_list_by_language(languages = ['en'], field="grounding_snippets"), # only keep english. can also be run on grounding_docs instead
        # remove_without_grounding_snippets,
        save_to_disk(json_output_stream, NDJsonExtractor()) # chose extractor here
    ]
    
    process(
        wiki_input_stream,
        base_generator = partial(generate_revision_pairs, max_bytes=max_bytes), # chose base generator here
        processors=processors
    )
    
    wiki_input_stream.close()
    json_output_stream.close()
    logging.info("Done with task %d" % args.index)
    logging.info(Profiled.summarize(processors))
