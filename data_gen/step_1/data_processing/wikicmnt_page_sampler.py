import argparse
import sys
import os
import logging
import datetime
import threading
import multiprocessing

from wiki_util import *
from wikicmnt_page_sampler_st import randSamplePage
from wikicmnt_extractor import WikiSampleTask

def workerProcess(task_id, dump_list, total_num, lock):
    logger = logging.getLogger(__name__)
    
    tasks = WikiSampleTask(dump_list, total_num, lock)
    threads = []
    for i in range(args.threads):
        t = threading.Thread(target=worker, args=(i, tasks))
        threads.append(t)
        t.start()

    logger.debug('Waiting for worker threads')
    main_thread = threading.currentThread()
    for t in threading.enumerate():
        if t is not main_thread:
            t.join()


#    while 1:
#        if len(dump_list) == 0:
#            break
#
#        dump_file = dump_list.pop(0)
#        logger.debug('Processing file: ' + str(dump_file))
#        output_file = args.output_path + 'enwiki-sample-' + \
#                os.path.basename(dump_file)[27:-4] + '.txt.bz2'
#        randSamplePage(task_id, dump_file, output_file, args.sample_ratio)

def worker(work_id, tasks):
    logger = logging.getLogger(__name__)

    logger.debug('Starting.')
    while 1:
        dump_file, cur_progress, total_num = tasks.assign_task()
        if not dump_file:
            break
        logger.debug('Assigned task (' + str(cur_progress) + '/' +
                str(total_num) + '): ' + str(dump_file))
        output_file = args.output_path + 'enwiki-sample-' + \
            os.path.basename(dump_file)[27:-4] + '.txt.bz2'
        randSamplePage(work_id, dump_file, output_file, args.sample_ratio)
    
    logger.debug('Exiting.')


def main():

    logger = logging.getLogger(__name__)

    manager = multiprocessing.Manager()

    dump_list = glob.glob(args.data_path + "*.bz2")

    if args.overwrite_existing:
        processed_files = glob.glob(args.output_path + "*.bz2")

        dump_names = [os.path.basename(f)[27:-4] for f in dump_list]
        processed_names = [os.path.basename(f)[14:-8] for f in processed_files]

        dump_list = [f for (f,dump_name) in\
                zip(dump_list, dump_names) if dump_name not in processed_names]

    dump_list = manager.list(dump_list)
#    print(len(dump_list))

    dump_num = len(dump_list)
    logger.debug('Sampling pages from ' + str(dump_num) + ' dump files')

#    p = multiprocessing.Pool()
#    p.map(workerProcess, dump_list)
   
#    task = WikiSampleTask(dump_list)
    lock = multiprocessing.Lock()
    jobs = []
    for i in range(0, args.processes):
        process = multiprocessing.Process(target=workerProcess, args=(i,\
            dump_list, dump_num, lock))
        jobs.append(process)
        process.start()

    logger.debug('Waiting for worker processes')
    for j in jobs:
        j.join()

#    task = WikiSampleTask(dump_list)
#    threads = []
#    for i in range(args.threads):
#        t = threading.Thread(target=worker, args=(i, task))
#        threads.append(t)
#        t.start()
#
#    logger.debug('Waiting for worker threads')
#    main_thread = threading.currentThread()
#    for t in threading.enumerate():
#        if t is not main_thread:
#            t.join()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Wiki Page Sampler')
    parser.add_argument('--data_path', type=str,
        default='/mnt/nlp-storage/data/raw/wikipedia-dumps/', help='the data\
        directory')
    parser.add_argument('--output_path', type=str,
        default='/mnt/nlp-storage/data/raw/wikipedia-subsample/', help='the\
        output directory')
    parser.add_argument('--sample_ratio', type=float, default=0.01)
    parser.add_argument('--threads', type=int, default=5)
    parser.add_argument('--processes', type=int, default=6)
    parser.add_argument('--overwrite_existing', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG,
            format='(%(processName)s) (%(threadName)s) (%(name)s) %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    start_time = datetime.datetime.now()
    main()
    time_elapsed = datetime.datetime.now() - start_time
    logger.debug('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

