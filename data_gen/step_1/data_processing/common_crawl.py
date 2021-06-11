import json, requests, gzip
import logging
import urllib
import re
from pywb.utils.canonicalize import canonicalize
from io import StringIO, BytesIO
from functools import partial
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dateparser import parse as parse_date

def silence_exceptions_with_none(f):
    def ignoring_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return None
    return ignoring_f

def get_first_or_none(iter):
    return next(filter(None, iter), None)

def first_result_parallel(func, iter, n_workers):
    """Executes a function func in parallel by n_workers until one result is not None, then returns this result preemptively.
        Returns (result, n_attempts)."""

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    attempts = 0
    batches = list(chunks(iter, n_workers))
    with ThreadPoolExecutor(max_workers = n_workers) as executor:
        for batch in batches:
            attempts += len(batch)
            result = get_first_or_none(executor.map(func, batch))
            if result: 
                return result, attempts

    return (None, attempts)

class CommonCrawlS3():

    # hide logging of urllib3
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    DefaultIndices = [
            'CC-MAIN-2020-16', 'CC-MAIN-2020-10', 'CC-MAIN-2020-05', 'CC-MAIN-2019-51', 'CC-MAIN-2019-47', 'CC-MAIN-2019-43',
            'CC-MAIN-2019-39', 'CC-MAIN-2019-35', 'CC-MAIN-2019-30', 'CC-MAIN-2019-26', 'CC-MAIN-2019-22', 'CC-MAIN-2019-18',
            'CC-MAIN-2019-13', 'CC-MAIN-2019-09', 'CC-MAIN-2019-04', 'CC-MAIN-2018-51', 'CC-MAIN-2018-47', 'CC-MAIN-2018-43',
            'CC-MAIN-2018-39', 'CC-MAIN-2018-34', 'CC-MAIN-2018-30', 'CC-MAIN-2018-26', 'CC-MAIN-2018-22', 'CC-MAIN-2018-17',
            'CC-MAIN-2018-13', 'CC-MAIN-2018-09', 'CC-MAIN-2018-05', 'CC-MAIN-2017-51', 'CC-MAIN-2017-47', 'CC-MAIN-2017-43',
            'CC-MAIN-2017-39', 'CC-MAIN-2017-34', 'CC-MAIN-2017-30', 'CC-MAIN-2017-26', 'CC-MAIN-2017-22', 'CC-MAIN-2017-17',
            'CC-MAIN-2017-13', 'CC-MAIN-2017-09', 'CC-MAIN-2017-04', 'CC-MAIN-2016-50', 'CC-MAIN-2016-44', 'CC-MAIN-2016-40',
            'CC-MAIN-2016-36', 'CC-MAIN-2016-30', 'CC-MAIN-2016-26', 'CC-MAIN-2016-22', 'CC-MAIN-2016-18', 'CC-MAIN-2016-07',
            'CC-MAIN-2015-48', 'CC-MAIN-2015-40', 'CC-MAIN-2015-35', 'CC-MAIN-2015-32', 'CC-MAIN-2015-27', 'CC-MAIN-2015-22',
            'CC-MAIN-2015-18', 'CC-MAIN-2015-14', 'CC-MAIN-2015-11', 'CC-MAIN-2015-06', 'CC-MAIN-2014-52', 'CC-MAIN-2014-49',
            'CC-MAIN-2014-42', 'CC-MAIN-2014-41', 'CC-MAIN-2014-35', 'CC-MAIN-2014-23', 'CC-MAIN-2014-15', 'CC-MAIN-2014-10',
            'CC-MAIN-2013-48', 'CC-MAIN-2013-20', 'CC-MAIN-2012', 'CC-MAIN-2009-2010', 'CC-MAIN-2008-2009' ]

    def __init__(self, indices = DefaultIndices, parallel_attempts = 5):
        self.indices = indices
        self.parallel_attempts = parallel_attempts
        self.cache = {} # TODO: limit memory of this cache
        self.call_count = 0
        self.fail_count = 0

    @staticmethod
    def fetch_html_from_s3_file(meta):
        offset, length = int(meta['offset']), int(meta['length'])
        offset_end = offset + length - 1
        prefix = 'https://commoncrawl.s3.amazonaws.com/'
        resp = requests.get(prefix + meta['filename'], headers={'Range': 'bytes={}-{}'.format(offset, offset_end)})
        with BytesIO(resp.content) as raw_data:
            with gzip.GzipFile(fileobj = raw_data) as f:
                data = f.read().decode("utf8")
                sections = data.strip().split('\r\n\r\n', 2)
                if len(sections) != 3:
                    # Number of sections != 3 usually occurs for non-200 status codes, i.e. no content was crawled
                    # logging.error("Received a non-standard data blob from CommonCrawl for page {}".format(str(meta)))
                    return None
                warc, header, response = sections
                if not "200 OK" in header:
                    # found the page in CC, but a non 200 HTTP code was crawled!
                    return None
                return response

    @silence_exceptions_with_none
    def get_html_from_index(self, index, url, closest_datetime_str = None):
        index_url = 'http://index.commoncrawl.org/' + index + "-index"
        params = [
            ('output', 'json'),
            ('limit', 1),
            ('url', url)]
        #queryparams = urllib.parse.urlencode(params)
        resp = requests.get(index_url, params)
        if resp.status_code != 200:
            return None
        lines = resp.content.decode("utf-8").strip().split('\n')
        references = (json.loads(x) for x in lines)
        #TODO: respect closest_datetime_str. Currently this feature is only supported in PreindexedCommonCrawlS3
        return get_first_or_none(map(CommonCrawlS3.fetch_html_from_s3_file, references))
        
    def get_html(self, url, closest_datetime_str = None):
        if url in self.cache:
            return self.cache[url]

        self.call_count += 1
        fetch = partial(self.get_html_from_index, url=url, closest_datetime_str=closest_datetime_str)
        result, attempts = first_result_parallel(fetch, self.indices, self.parallel_attempts)
        self.cache[url] = result
        if result is None: self.fail_count += 1
        outcome_word = "SUCCEEDED" if result else "FAILED"
        success_count =  self.call_count - self.fail_count
        success_percent = success_count * 100 / self.call_count
        logging.debug("{} fetching grounding document for {} after trying {} indices. Succeeded: {}/{} ({:.1f}%)"
            .format(outcome_word, url, attempts, success_count, self.call_count, success_percent))
        return result

def sort_by_closest(metas, closest_datetime):
    """Given a list of index snapshot metadata, and a datetime reference, sorts metadata in-place by temporally closest to reference time"""
    
    closest_datetime = closest_datetime.replace(tzinfo=None)
    def delta_to_approx_crawl_date(meta):
        m = re.search(r"CC-MAIN-(\d\d\d\d)(\d\d)(\d\d)", meta['filename']) # use filename as proxy for crawl time
        if m:
            year, month, day = m.group(1), m.group(2), m.group(3)
            crawl_time = datetime(int(year), int(month), int(day), tzinfo=None)
            delta = abs(closest_datetime - crawl_time)
            return delta
    try:
        metas.sort(key=delta_to_approx_crawl_date)
        return True
    except Exception as e:
        logging.error("Could not sort CommonCrawl snapshots by closest reference time: " + str(e))
        return False

class PreindexedCommonCrawlS3:
    """A variant of CommanCrawlS3 that uses a pre-joined index file to determine the file / offset to download"""
    
    def __init__(self, index_file):
        self.meta_index = defaultdict(list)
        with open(index_file, encoding="utf8") as f:
            for line in f:
                try:
                    canonical, _, meta = line.strip().split("\t")
                    self.meta_index[canonical].append(meta)
                except Exception as e:
                    logging.error("Malformed line in index file {}: {}. Exception: {}".format(index_file, line, str(e)))

    def get_html(self, url, closest_datetime_str = None):
        canonical = canonicalize(url)
        metas = self.meta_index.get(canonical)
        if not metas: return None
        metas = [json.loads(m) for m in metas]
        metas = [m for m in metas if m['status'] == "200"]
        if len(metas) > 1 and closest_datetime_str and sort_by_closest(metas, parse_date(closest_datetime_str)):
            pass # successfully sorted metas by reference time
        else:
            metas = sorted(metas, key = lambda m: m['filename'], reverse=True)
    
        html = get_first_or_none(map(CommonCrawlS3.fetch_html_from_s3_file, metas))
        return html

if __name__ == "__main__":
    from grounding_helpers import extract_text_bs4
    cc = CommonCrawlS3()
    #cc = PreindexedCommonCrawlS3("grounding_index.txt")
    html = cc.get_html("https://en.wikipedia.org/wiki/Barack_Obama")
    text = extract_text_bs4(html)
    print(text)