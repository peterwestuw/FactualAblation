from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import logging
import os
import threading
import urllib.request
from datetime import datetime

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient



parser = argparse.ArgumentParser()
parser.add_argument('--container_name', type=str, default='wikipedia-data')

args = parser.parse_args()


def existFile(data_path, cur_file):
    exist_file_list = [b.name for b in container_client.list_blobs()]
    exist_file_list = filter(lambda p: data_path in p, exist_file_list)
    exist_file_names = [os.path.basename(i) for i in exist_file_list]
    cur_file_name = os.path.basename(cur_file)
    if cur_file_name in exist_file_names:
        return True
    return False

from wiki_dump_download import download, main, WikiDumpTask, worker


if __name__ == '__main__':

    # Azure Blob Storage Connection
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(args.container_name)
    
    print(existFile('raw/', 'dumpstatus.json'))
