#!/usr/bin/env python

from dotenv import load_dotenv
load_dotenv()

import sys
import datetime
import bz2
import logging
import json
import argparse
import os
import io

from wiki_util import *
from wiki_dump_download import existFile

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient


def printSample(task_id, sample_count, revision_count, page_title, sect_title, comment, diff_url, parent_tokens,
                target_tokens, origin_diff, target_diff, delimitor='^'):
    logger=logging.getLogger(__name__)
    
    # print the sampled revision in excel format
    revision_info = '[' + str(len(parent_tokens)) + '|' + str(len(target_tokens)) + ']'
    origin_diff_tokens = '[' + (
        str(origin_diff[0]) if len(origin_diff) == 1 else ','.join([str(i) for i in origin_diff])) + ']'
    target_diff_tokens = '[' + (
        str(target_diff[0]) if len(target_diff) == 1 else ','.join([str(i) for i in target_diff])) + ']'
    # print(sample_count, '/', revision_count, delimitor, page_title, delimitor, sect_title, delimitor, comment, delimitor, 
    #       diff_url, delimitor, revision_info, delimitor, origin_diff_tokens, delimitor, target_diff_tokens, sep='')                    
    logger.info("[" + str(task_id) + "] " + str(sample_count) + '/' + str(
        revision_count) + delimitor + page_title + delimitor + sect_title +
                 delimitor + comment + delimitor + diff_url + delimitor + revision_info + delimitor + origin_diff_tokens + delimitor + target_diff_tokens)


def randSampleRev(task_id, dump_file, output_file, sample_ratio, min_cmnt_length, ctx_window, negative_cmnt_num=10,
                  negative_edit_num=10, count_revision_only=False,
                  MIN_COMMENT_SIZE=20, max_page_count=None, azure=False,
                  blob_service_client=None, container_name=None):
    # if os.path.exists(output_file):
    #     print("Output file already exists. Please remove first so I don't destroy your stuff please")
    #     sys.exit(1)

    logger=logging.getLogger(__name__)
    
    if not azure:
        json_file = open(output_file, "w", buffering=1, encoding='utf-8')
    else:
        json_file = io.StringIO()

    start_time = datetime.datetime.now()
    if not azure:
        wiki_file = bz2.open(dump_file, "rt", encoding='utf-8')
    else:
        wiki_file = blob_service_client.get_blob_client(container=container_name,
                blob=dump_file)
#        byte_stream = io.BytesIO()
#        blob_client.download_blob().download_to_stream(byte_stream)
#        byte_stream.seek(0)
#        wiki_file = bz2.open(byte_stream, 'rt', encoding='utf-8')
    # template = '{"revision_id": %s, "diff_url": %s, "parent_id": %s, "timestamp": %s, "page_title": %s, "user_name": %s, \
    #             "user_id": %s, "user_ip": %s, "comment": %s, "parent_text": %s, "text": %s, "diff_tokens": %s}\n'
    sample_count = 0
    revision_count = 0
    page_count = 0
    revisions = {}
    page_revision_count = 0
    revisions_per_page = {}

    # time_elapsed = datetime.datetime.now() - start_time
    # print("=== ", i, "/", file_num, ' === ', revision_count, " revisions extracted.", ' Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed), sep='')

    sample_parent_id = None
    sample_parent_text = None
    page_comment_list = []
    prev_page_title = '' 

    try:
        for page_title, page_id, revision in split_records(wiki_file, azure):
            revision_count += 1
           
            #if count_revision_only:
            #    if revision_count % 1000 == 0:
            #        logging.info("= revision" + str(revision_count) + " =")
            #    continue

            # fields 
            rev_id, parent_id, timestamp, username, userid, userip, comment, text = extract_data(revision)

            if count_revision_only:
                if prev_page_title != page_title:
                    revisions_per_page[prev_page_title] = page_revision_count
                    page_revision_count = 0
                    page_count += 1

                page_revision_count += 1

                comment = cleanCmntText(comment)
                # extract the section title and the comment without section info
                sect_title, comment = extractSectionTitle(comment)

                meta = {"comment_text":comment,
                        "text_length":len(text), "parent_id":parent_id,
                        "section_title":sect_title, "page_title":page_title}
                        
                revisions[rev_id] = meta

                if revision_count % 1000 == 0:
                    logging.info("= revision" + str(revision_count) + " =")
                continue
           
           # if the length of comment is less than MIN_COMMENT SIZE (default 20), skip the revision directly.
            if len(comment) < MIN_COMMENT_SIZE:
                continue

            # clean the comment text
            comment = cleanCmntText(comment)
            # extract the section title and the comment without section info
            sect_title, comment = extractSectionTitle(comment)

            # skip the revision if no section title included or the length is too short.
            if not sect_title or len(comment) < MIN_COMMENT_SIZE:
                continue

            # store the comments
            if prev_page_title != page_title:
                # TEMP
                print('CURRENT PAGE TITLE: {}'.format(page_title))
                
                page_count += 1
                prev_page_title = page_title
                page_comment_list.clear()
                if max_page_count:
                    if page_count > max_page_count: break

            _, comment_tokens = tokenizeText(comment)
            if checkComment(comment, comment_tokens, min_cmnt_length):
                # page_comment_list.append(comment)
                page_comment_list.append(comment)

            # write the sampled revision to json file
            # Skip the line if it is not satisfied with some criteria
            if sample_parent_id == parent_id:
                # do sample
                diff_url = 'https://en.wikipedia.org/w/index.php?title=' + page_title.replace(" ",
                                                                                              '%20') + '&type=revision&diff=' + rev_id + '&oldid=' + parent_id
                # print(diff_url)

                # check whether the comment is appropriate by some criteria
                # check_comment(comment, length_only=True)
                try:
                    src_text = extractSectionText(sample_parent_text, sect_title)
                    tgt_text = extractSectionText(text, sect_title)
                except:
                    print("ERROR-RegularExpression:", sample_parent_text, text, " Skip!!")
                    # skip the revision if any exception happens
                    continue

                # clean the wiki text
                src_text = cleanWikiText(src_text)
                tgt_text = cleanWikiText(tgt_text)

                if (src_text and tgt_text) and (len(src_text) < 1000000 and len(tgt_text) < 1000000):

                    # tokenization
                    src_sents, src_tokens = tokenizeText(src_text)
                    tgt_sents, tgt_tokens = tokenizeText(tgt_text)

                    # extract the offset of the changed tokens in both src and tgt
                    src_token_diff, tgt_token_diff = diffRevision(src_tokens, tgt_tokens)

                    if len(src_token_diff) == 0 and len(tgt_token_diff) == 0:
                        continue

                    if (len(src_token_diff) > 0 and src_token_diff[0] < 0) or (
                            len(tgt_token_diff) > 0 and tgt_token_diff[0] < 0):
                        continue

                    if src_sents == None or tgt_sents == None:
                        continue

                    src_ctx_tokens, src_action = extContext(src_tokens, src_token_diff, ctx_window)
                    tgt_ctx_tokens, tgt_action = extContext(tgt_tokens, tgt_token_diff, ctx_window)

                    # src_sent_diff = findSentDiff(src_sents, src_tokens, src_token_diff)
                    tgt_sent_diff = findSentDiff(tgt_sents, tgt_tokens, tgt_token_diff)

                    # randomly sample the negative comments
                    if negative_cmnt_num > len(page_comment_list) - 1:
                        neg_cmnt_idx = range(len(page_comment_list) - 1)
                    else:
                        neg_cmnt_idx = random.sample(range(len(page_comment_list) - 1), negative_cmnt_num)
                    neg_comments = [page_comment_list[i] for i in neg_cmnt_idx]

                    # generate the positive edits
                    pos_edits = [tgt_sents[i] for i in tgt_sent_diff]

                    # generate negative edits
                    neg_edits_idx = [i for i in range(len(tgt_sents)) if i not in tgt_sent_diff]
                    if negative_edit_num > len(neg_edits_idx):
                        sampled_neg_edits_idx = neg_edits_idx
                    else:
                        sampled_neg_edits_idx = random.sample(neg_edits_idx, negative_edit_num)
                    neg_edits = [tgt_sents[i] for i in sampled_neg_edits_idx]

                    if (len(src_token_diff) > 0 or len(tgt_token_diff) > 0):
                        json_dict = {"revision_id": rev_id, "parent_id": parent_id, "timestamp": timestamp, \
                                     "diff_url": diff_url, "page_title": page_title, \
                                     "comment": comment, "src_token": src_ctx_tokens, "src_action": src_action, \
                                     "tgt_token": tgt_ctx_tokens, "tgt_action": tgt_action, \
                                     "neg_cmnts": neg_comments, "neg_edits": neg_edits, "pos_edits": pos_edits
                                     }

                        json_str = json.dumps(json_dict,
                                              indent=None, sort_keys=False,
                                              separators=(',', ': '), ensure_ascii=False)
                        json_file.write(json_str + '\n')
                        sample_count += 1

                        printSample(task_id, sample_count, revision_count, page_title, sect_title, comment, diff_url,
                                    src_tokens, tgt_tokens, src_token_diff, tgt_token_diff)

                # filterRevision(comment, diff_list):
                # if sample_parent_id != parent_id:
                #     logging.debug("ALERT: Parent Id Missing" + str(sample_parent_id) + "/" + str(parent_id) + "--- DATA MIGHT BE CORRUPTED!!")

            # decide to sample next
            if sampleNext(sample_ratio):
                sample_parent_id = rev_id
                sample_parent_text = text
            else:
                sample_parent_id = None
                sample_parent_text = None

            # if revision_count % 1000 == 0:
            #     print("Finished ", str(revision_count))
    finally:
        time_elapsed = datetime.datetime.now() - start_time
        logger.debug("=== " + str(sample_count) + " revisions sampled in total " + str(revision_count) + " revisions. " \
                      + 'Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed) + ' ===')
        if count_revision_only:
            json_dict = {"revision_count": revision_count, "time_elapsed":
                    "(hh:mm:ss.ms) {}".format(time_elapsed),
                         "page_count": page_count,
                         "revisions_per_page": revisions_per_page,
                         "revision": revisions}
            json_str = json.dumps(json_dict,
                                  indent=None, sort_keys=False,
                                  separators=(',', ': '), ensure_ascii=False)
            json_file.write(json_str + '\n')
            
        if not azure: 
            json_file.close()
        else:
            json_file.seek(0)
            data = json_file.read().encode('utf-8')
            blob_client = blob_service_client.get_blob_client(container=container_name,
                blob=output_file)
            container_client = blob_service_client.get_container_client(container_name)
            if existFile('processed/', output_file, container_client, azure=2):
                blob_client.delete_blob()
            blob_client.upload_blob(data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--azure', action='store_true')
    parser.add_argument('--container_name', type=str, default='wikipedia-data')
    parser.add_argument('--dump_file', type=str)
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                    format='(%(threadName)s) (%(name)s) %(message)s',
                    )

    #Azure connection
    if args.azure:
        connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    
    randSampleRev(1, args.dump_file, args.output_file, 1, 10, 5,
            count_revision_only=True, azure=args.azure)
