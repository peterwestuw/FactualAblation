"""
Runs a partial pipeline from a set of NDJson checkpoint files
"""

import os, io, argparse
import re, csv
import logging
import urllib
import json

from functools import partial
from generator_chaining import process
from profiling import Profiled
from custom_filters import *
from custom_extractors import *

def scriptdir(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def convert_to_cstr_format():
    from nltk.corpus import stopwords
    stopwords = set(stopwords.words('english'))
    
    frequency = {}
    with open(scriptdir('unigram_freq.csv'), encoding="utf-8") as f:
        frequency = dict((x['word'], int(x['count'])) for x in csv.DictReader(f))

    counter = 0
    def convert_to_cstr_format(instance):
        nonlocal counter
        counter += 1

        context_sentences = [
            "<t{}> ".format(idx + 1) + " ".join(tokens)
            for (idx, tokens) in enumerate(instance["left_context"])
        ]

        context = " ".join(context_sentences)
        lines = [
            str(counter), 
            context,
            instance['tgt_text'].lower()
        ]

        comment_tokens = re.findall(r"\w+", instance["comment_text"])
        comment_tokens_set = set(t.lower() for t in comment_tokens).difference(stopwords)
        groundings = instance["grounding_snippets"].lower().split(" ;; ")

        for grounding in groundings:
            control = "n/a"
            grounding_tokens = list(re.findall(r"\w+", grounding))
            grounding_token_set = set(grounding_tokens)
            intersect_tokens = list(comment_tokens_set.intersection(grounding_token_set))
            if len(intersect_tokens):
                control = min(intersect_tokens, key = lambda t: frequency.get(t, 1000000))
            else:
                # fallback, in case comment text and grounding have no overlapping token:
                tgt_text_tokens = set(t.lower() for t in instance['tgt_tokens'])
                intersect_tokens = tgt_text_tokens.difference(stopwords).intersection(grounding_token_set)
                if len(intersect_tokens):
                    control = min(intersect_tokens, key = lambda t: frequency.get(t, 1000000))
                else:
                    # fallback in case there is also no overlap between grounding and target text
                    pass
            
            lines.append("<p>" + grounding)
            lines.append(control)
            lines.append("")
        
        result = "\n".join(lines)
        yield result
    return convert_to_cstr_format

def ndjson_generator(input_stream):
    for line in input_stream:
        object = json.loads(line)
        yield object

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-path', type=str, default="./data/in/", help='the input directory')
    parser.add_argument('--output-path', type=str, default="./data/out/", help='the output directory')
    parser.add_argument('--index', default=None, type=int, help='the index of the file in the input path to process')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='(%(threadName)s) %(message)s')
    output_file = args.output_path
    input_file = args.input_path
    if args.index:
        input_file = os.path.join(args.input_path, "{}.json".format(args.index))
        output_file = os.path.join(args.output_path, "{}.txt".format(args.index))
    
    input_stream = open(input_file, "rt", encoding='utf-8')
    output_tream = open(output_file, "w", buffering=1, encoding='utf-8')

    ### chose processing and filtering steps here:
    processors = [
        convert_to_cstr_format(),
        save_to_disk(output_tream, StringExtractor()), # chose extractor here
    ]
    
    process(
        input_stream,
        base_generator = ndjson_generator,
        processors=processors
    )
    
    input_stream.close()
    output_tream.close()
    logging.info("Done with task {}".format(args.index))
    logging.info(Profiled.summarize(processors))