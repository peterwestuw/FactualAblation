import os, io, argparse, logging, json, gzip
import urllib.request
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

CollectionInfo = "https://index.commoncrawl.org/collinfo.json"
PathPattern = "https://commoncrawl.s3.amazonaws.com/crawl-data/{}/cc-index.paths.gz"
IndexPattern = "https://commoncrawl.s3.amazonaws.com/{}"

def read_from_path_url(path_gz_url):
    try:
        response = urllib.request.urlopen(path_gz_url)
        content = gzip.decompress(response.read())
        lines = content.decode("utf-8").splitlines()
        paths = [line.strip() for line in lines if "cdx" in line]
        urls = [IndexPattern.format(path) for path in paths]
        return urls
    except Exception as e:
        logging.error("Didn't work for {}: {}".format(path_gz_url, str(e)))
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-file', type=str, default="cc-indices.txt", help='the output file containing one index url per line')
    args = parser.parse_args()

    indices = json.loads(urllib.request.urlopen(CollectionInfo).read())
    path_files = [PathPattern.format(x["id"]) for x in indices]
    urls = (url for path_file in path_files for url in read_from_path_url(path_file))
    with open(args.output_file, 'w', encoding="utf-8") as out:
        for url in urls:
            out.write(url + "\n")

