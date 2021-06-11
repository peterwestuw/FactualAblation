from bs4 import BeautifulSoup
from itertools import accumulate
from operator import itemgetter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

def extract_text_bs4(html):
    return BeautifulSoup(html).get_text()

def extract_overlapping_tokens(target_length, reference_tokens, grounding_doc_tokens):
    """Extracts a subslice from a list of grounding_doc_tokens of give target_length that
    has the most occurences of tokens from a given set of reference_tokens"""


    reference_tokens = set(reference_tokens)

    if len(grounding_doc_tokens) < target_length:
        # if the grounding document is shorter than target length, return it unaltered
        overlap = len(reference_tokens.intersection(set(grounding_doc_tokens)))
        return grounding_doc_tokens, overlap

    # compute for each ending index of the subslice the number of tokens in the reference token set..
    is_in_ref = [int(t in reference_tokens) for t in grounding_doc_tokens]
    in_ref_counts = list(accumulate(is_in_ref))
    count_here = [in_ref_counts[end] - in_ref_counts[end - target_length] for end in range(target_length, len(in_ref_counts))]

    if not len(count_here):
        overlap = len(reference_tokens.intersection(set(grounding_doc_tokens)))
        return grounding_doc_tokens, overlap

    # determine the possible indices of slices with a maximum count of reference tokenso
    best_count = max(count_here)
    best_ending_indices = [idx + target_length for (idx, c) in enumerate(count_here) if c == best_count]

    # select the slice in the middle of the candidate positions (equal padding left and right)
    best_end_idx = best_ending_indices[len(best_ending_indices) // 2]
    start_idx = max(0, best_end_idx - target_length)
    subslice = grounding_doc_tokens[start_idx:best_end_idx]
    #print("best count = {}".format(best_count))
    #print("sample: {}".format(" ".join(subslice[:100])))
    return subslice, best_count

if __name__ == '__main__':
    from common_crawl import CommonCrawlS3
    cc = CommonCrawlS3()
    html = cc.get_html("https://en.wikipedia.org/wiki/Barack_Obama")
    text = extract_text_bs4(html)
    tokens = word_tokenize(text)

    # example: extract the 100 token subslice containing the most stopwords:
    overlap = extract_overlapping_tokens(100, stopwords.words('english'), tokens)
    print(overlap)