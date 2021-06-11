"""
Contains self-contained isolated filters/processors implemented as python generators
"""

import logging, re
from profiling import Profiled

def page_id_filter(accepted_ids):
    def generate(meta):
        if meta['page_id'] in accepted_ids:
            yield meta
    return generate

def has_comment(meta):
    if meta["comment_text"]: yield meta

def comment_length(min_len, max_len):
    @Profiled.generator
    def comment_length(meta):
        clen = len(meta["comment_text"])
        if clen >= min_len and clen < max_len:
            yield meta
    return comment_length

def comment_token_length(min_len, max_len):
    @Profiled.generator
    def comment_token_length(meta):
        approx_tokens = meta["comment_text"].split(" ")
        clen = len(approx_tokens)
        if clen >= min_len and clen < max_len:
            yield meta
    return comment_token_length


def comment_blocklist_filter(exclude_words = ["[[Project:AWB|AWB]]", "[[Project:AutoWikiBrowser|AWB]]", "Undid revision"]):
    @Profiled.generator
    def comment_blocklist_filter(meta):
        comment = meta["comment_text"]
        if not any(word in comment for word in exclude_words):
            yield meta
    return comment_blocklist_filter

def clean_newlines(fields=["src_text", "tgt_text"]):
    def clean(text):
        text = re.sub(r"[\r\n]+", "\n", text) # newlines
        text = re.sub(r"[\t \xa0]+", " ", text) # various spaces but not newlines (can't use \s and python re doesn't have \h)
        return text.strip()
    
    @Profiled.generator
    def clean_newlines(instance):
        for field in fields:
            if isinstance(instance[field], str):
                instance[field] = clean(instance[field])
            if isinstance(instance[field], list):
                instance[field] = list(map(clean, instance[field]))
        yield instance
    return clean_newlines

def text_length(min_len, max_len):
    @Profiled.generator
    def text_length(instance):
        len_src = len(instance["src_text"])
        len_tgt = len(instance["tgt_text"])
        if len_src >= min_len and len_tgt >= min_len and len_src < max_len and len_tgt < max_len:
            yield instance
    return text_length

def exclude_page_types(excludes_prefixes = ["Talk:"]):
    @Profiled.generator
    def exclude_page_types(meta):
        has_any_prefix = any(prefix in meta["page_title"] for prefix in excludes_prefixes)
        if not has_any_prefix:
            yield meta
    return exclude_page_types

@Profiled.generator
def has_section_title(instance):
    if instance['section_title']:
        yield instance

def is_human_edit(instance):
    raise NotImplementedError

def canonize_grounding():
    from pywb.utils.canonicalize import canonicalize
    
    @Profiled.generator
    def canonize_grounding(instance):
        if "grounding_urls" in instance:
            try:
                instance["grounding_urls"] = [url.lower() for url in instance["grounding_urls"]]
                instance["grounding_canonical_urls"] = [canonicalize(url) for url in instance["grounding_urls"]]
            except Exception as e:
                logging.error("Could not canonize grounding URLs: " + str(e))
                return
        yield instance

    return canonize_grounding

def has_urls_in_text(look_in_src = True, look_in_tgt = True):
    """Filters instances to contain at least one url and extracts a preliminary 'grounding_urls' field"""

    @Profiled.generator
    def has_urls_in_text(instance):
        url_set = set()
        if look_in_src:
            url_set.update(instance["src_urls"])
        if look_in_tgt:
            url_set.update(instance["tgt_urls"])
        if len(url_set) > 0:
            # preliminary set of grounding urls. May be filtered/overridden later on
            instance["grounding_urls"] = url_set
            yield instance

    return has_urls_in_text


def has_grounding():
    """Filters instances that have a grounding_url set at this stage"""

    @Profiled.generator
    def has_grounding(instance):
        if instance.get("grounding_urls"):
            yield instance

    return has_grounding


def restrict_grounding_to_max_distance(max_token_distance, url_replacement_str = "URL"):
    from heapq import nsmallest
    
    @Profiled.generator
    def restrict_grounding_to_max_distance(instance):
        try:
            src_indices = [i for i,token in enumerate(instance['src_tokens']) if token == url_replacement_str]
            tgt_indices = [i for i,token in enumerate(instance['tgt_tokens']) if token == url_replacement_str]
            tgt_edit_indices = set(instance['tgt_token_diff'])
            src_edit_indices = set(instance['src_token_diff'])

            def t_dist(token_index, index_list):
                closest = nsmallest(1, index_list, key=lambda x: abs(x-token_index))
                if len(closest) < 1: return 100000000
                dist = abs(closest[0] - token_index)
                return dist

            
            src_urls_with_dist = [(url_index, t_dist(token_index, src_edit_indices)) for url_index, token_index in enumerate(src_indices)]
            tgt_urls_with_dist = [(url_index, t_dist(token_index, tgt_edit_indices)) for url_index, token_index in enumerate(tgt_indices)]

            instance["url_src_distances"] = src_urls_with_dist
            instance["url_tgt_distances"] = tgt_urls_with_dist
            tgt_url_respecting_dist = [instance["tgt_urls"][url_index] for url_index, token_dist in tgt_urls_with_dist if token_dist < max_token_distance]
            src_url_respecting_dist = [instance["src_urls"][url_index] for url_index, token_dist in src_urls_with_dist if token_dist < max_token_distance]

            grounding_set = set(tgt_url_respecting_dist).union(set(src_url_respecting_dist))
            instance["grounding_urls"] = list(grounding_set)
            yield instance
        except Exception as e:
            logging.error("Could not restrict grounding urls to max token distance: " + str(e))

    return restrict_grounding_to_max_distance

def clean_urls(replacement='URL'):
    """Replaces URLs in the text with replacement string, and keeps list of URLs seperately from text"""
    url_regex = re.compile(r"https?://[^\s|\]]+", flags=re.MULTILINE)

    @Profiled.generator
    def clean_urls(instance):
        instance["src_urls"] = url_regex.findall(instance["src_text"])
        instance["tgt_urls"] = url_regex.findall(instance["tgt_text"])
        instance["original_src_text"] = str(instance["src_text"])
        instance["original_tgt_text"] = str(instance["tgt_text"])
        instance["src_text"] = url_regex.sub(replacement, instance["src_text"])
        instance["tgt_text"] = url_regex.sub(replacement, instance["tgt_text"])
        yield instance

    return clean_urls

def grounding_domain_whitelist(whitelist=[], file=None):
    if file:
        with open(file, "r", encoding="utf-8") as f:
            whitelist = [x.strip() for x in f.readlines()]
    # whitelist = ["://" + x for x in whitelist]

    @Profiled.generator
    def grounding_domain_whitelist(instance):
        def filter_urls(list_of_urls):
            return [url for url in list_of_urls if any(domain in url for domain in whitelist)]

        url_fields = ["grounding_urls", "src_urls", "tgt_urls"]
        for field in url_fields:
            instance[field] = filter_urls(instance[field])

        num_urls = sum([len(instance[f]) for f in url_fields])
        if num_urls > 0:
            yield instance

    return grounding_domain_whitelist


def extract_common_crawl_groundings(prejoined_index_file = None):
    from common_crawl import CommonCrawlS3, PreindexedCommonCrawlS3
    from grounding_helpers import extract_text_bs4
    cc = PreindexedCommonCrawlS3(prejoined_index_file) if prejoined_index_file else CommonCrawlS3()

    @Profiled.generator
    def extract_common_crawl_groundings(instance):
        def download_grounding(url):
            try:
                html = cc.get_html(url, closest_datetime_str=instance.get('timestamp'))
                if not html: return None
                text = extract_text_bs4(html)
                return text

            except Exception as e:
                logging.error("Error fetching grounding for {}: {}".format(url, str(e)))
                return None
        
        grounding_docs = list(filter(None, map(download_grounding, instance["grounding_urls"])))
        instance["grounding_docs"] = grounding_docs
        yield instance
    
    return extract_common_crawl_groundings


def filter_list_by_language(languages=['en'], field="grounding_docs"):
    import langid

    def matches(text):
        l, _ = langid.classify(text[:100000]) # limit max text length
        return l in languages # only use highest scored prediction

    @Profiled.generator
    def filter_list_by_language(instance):
        
        instance[field] = list(filter(matches, instance[field]))
        yield instance

    return filter_list_by_language

def extract_grounding_snippet(target_length, min_overlap_tokens):
    from grounding_helpers import extract_overlapping_tokens
    from nltk import word_tokenize
    from nltk.corpus import words
    from nltk.corpus import stopwords

    stopwords = stopwords.words('english')
    #english_words = set(words.words('en'))
    #punctuation = set("{}[]()<>.,;!?+-'’‘\\/`")
    #subtract_set = punctuation.union(stopwords)

    @Profiled.generator
    def extract_grounding_snippet(instance):
        target_tokens = set(instance["tgt_tokens"])
        approx_tokens = set(instance["comment_text"].split(" "))
        reference_tokens = target_tokens.union(approx_tokens)
        reference_tokens = set(r for r in reference_tokens if re.match(r"\w+", r))
        reference_tokens = reference_tokens.difference(stopwords)

        def extract_snippet(text):
            tokens = word_tokenize(text)
            overlap, overlap_count = extract_overlapping_tokens(target_length, reference_tokens, tokens)
            if overlap_count < min_overlap_tokens:
                return None
            overlap_text = " ".join(overlap) # dirty hack
            return overlap_text

        snippets = list(filter(None, map(extract_snippet, instance["grounding_docs"])))
        instance["grounding_snippets"] = snippets
        yield instance
    return extract_grounding_snippet

@Profiled.generator
def concat_grounding_snippets(instance):
     # dirty hack: concat all grounding snippets into one. keept for data compatatibility, to be removed in v2
    instance["grounding_snippets"] = " ;; ".join(instance["grounding_snippets"]) if len(instance["grounding_snippets"]) else None

@Profiled.generator
def remove_without_grounding_snippets(instance):
    """Removes all documents lacking a grounding snippet"""
    if instance.get("grounding_snippets"):
        yield instance

@Profiled.generator
def remove_without_grounding_docs(instance):
    """Removes all documents lacking a grounding document"""
    if instance.get("grounding_docs"):
        yield instance