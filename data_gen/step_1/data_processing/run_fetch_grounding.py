"""
Downloads (on-demand) a range of dump files and processes them according to filters and processors specified below.
"""

import os, io, argparse
import logging
import urllib.request
import json
import gzip
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from tqdm import tqdm
from profiling import Profiled
from generator_chaining import chain_generators
from custom_extractors import save_to_disk
from custom_extractors import StringExtractor

def iterate_lines(stream):
    for line in stream: 
        yield line.strip()

def read_lines(file, col = list):
    with open(file, encoding="utf8") as f:
        return col(iterate_lines(f))

def get_cc_url(index_file, start_idx, end_idx):
    lines = list(read_lines(index_file))
    try: return lines[start_idx:end_idx]
    except: return None

def download_cc_on_demand(url, temp_folder):
    # Hack for HPC: cert verification issues
    parts = url.split("/")
    index_file = os.path.join(temp_folder, parts[-3], parts[-1])
    exists = os.path.exists(index_file)
    if not exists:
        urllib.request.urlretrieve(url, index_file)
    else:
        logging.info("File Exists, Skip: " + url)
    return index_file

def stream_from_index_files(index_files):
    for index_file in index_files:
        logging.info("Fetching index {} to memory..".format(index_file))
        response = urllib.request.urlopen(index_file)
        content = gzip.decompress(response.read())
        lines = content.decode("utf-8").splitlines()
        logging.info("Iterating through {}..".format(index_file))
        for line in lines:
            yield line.strip()

def scriptdir(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

def filter_to_url_set(grounding_file):
    urls = read_lines(grounding_file, col=set)
    
    @Profiled.generator
    def filter_to_url_set(line):
        _, _, obj = line.split(" ", 2)
        obj = json.loads(obj)
        url = obj["url"]
        if url in urls:
            yield line
    return filter_to_url_set

def process(base_generator, processors):
    """Iterates through base_generator, then chains processor steps in processors"""
    iterable = tqdm(base_generator, "baseline generator", mininterval=3.0)
    results = chain_generators(iterable, processors)
    for _ in tqdm(results, "final results", mininterval=3.0):
        Profiled.total_count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grounding-urls', type=str, help='file containing grounding urls to join')
    parser.add_argument('--cc-index-file', type=str, default='./data/cc-indices.txt')
    parser.add_argument('--index', type=int, help='the index of the file (within the common-crawl index file) to download and process')
    parser.add_argument('--range-factor', type=int, default=1, help='if not equal 1 interprets index as the start of a (range index*factor, index*factor + factor)')
    ##parser.add_argument('--temp-path', type=str, default="./data/cc-index/", help='the temp / data directory, used to download wiki dump files')
    parser.add_argument('--output-path', type=str, default="./data/out/", help='the output directory')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG, format='(%(threadName)s) %(message)s')

    start_idx = args.index * args.range_factor
    end_idx = start_idx + args.range_factor
    logging.info("Determining index joining task for index {}, factor {}, resulting in task range ({}, {})".format(
        args.index, args.range_factor, start_idx, end_idx))
    index_urls = get_cc_url(args.cc_index_file, start_idx, end_idx)
    logging.info("Working on index urls: " + ",".join(index_urls))

    logging.info("Loading grounding url set from " + args.grounding_urls)
    filter_urls = filter_to_url_set(args.grounding_urls)

    ## Old method: fetch to file first..
    ##index_file = download_cc_on_demand(index_url, args.temp_path)
    ##logging.info("Dumped to " + index_file + " processing to " + output_file)
    ##cc_index_input_stream = gzip.open(index_file, "rt", encoding='utf-8')

    output_file = os.path.join(args.output_path, "{}.ccidx".format(args.index))
    output_stream = open(output_file, "w", buffering=1, encoding='utf-8')

    processors = [
            filter_urls,
            save_to_disk(output_stream, StringExtractor())
    ]
    process(
        base_generator = stream_from_index_files(index_urls),
        processors = processors        
    )

    ##cc_index_input_stream.close()
    output_stream.close()
    logging.info("Done with task %d" % args.index)
    logging.info(Profiled.summarize(processors))
