import sys
import datetime
import bz2
import logging
import argparse
import os
import io

from wiki_util import *

def randSamplePage(task_id, dump_file, output_file, sample_ratio):

    logger = logging.getLogger(__name__)

    out_file = bz2.open(output_file, 'wt', encoding='utf-8')

    start_time = datetime.datetime.now()
    wiki_file = bz2.open(dump_file, 'rt', encoding='utf-8')

    sample_count = 0
    page_count = 0
    cur_page_title = ''

    try:
        #for page_title, revision in split_records(wiki_file):
        #    if page_title != cur_page_title:
        #        logger.info(page_title)
        #        cur_page_title = page_title

        for i, page in enumerate(split_pages(wiki_file)):
#            if i >= 5: break

            page_count += 1
            if (page_count+1) % 1000 == 0:
                logger.info('Page {}'.format(page_count))

            #decide to sample
            if sampleNext(sample_ratio):
                out_file.write(page + '\n')
                sample_count += 1
    finally:
        time_elapsed = datetime.datetime.now() - start_time
        logger.info("===" + str(sample_count) + " pages sampled. Time elapsed\
                (hh:mm:ss.ms) {}".format(time_elapsed) + ' ===')
        out_file.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_file', type=str,
        default='/mnt/nlp-storage/data/raw/wikipedia-dumps/enwiki-20200201-pages-meta-history10.xml-p2336425p2366571.bz2')
    parser.add_argument('--output_file', type=str,
        default='/mnt/nlp-storage/data/raw/wikipedia-subsample/test.txt.bz2')
    parser.add_argument('--sample-ratio', type=float, default=0)
    args = parser.parse_args()

    logging.basicConfig(level = logging.INFO,
            format='(%(threadName)s) (%(name)s) %(message)s',
            )

    randSamplePage(1, args.dump_file, args.output_file, args.sample_ratio)
    
