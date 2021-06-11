# Code taken and modified from Wikiextractor

from WikiExtractor import *

# tags to ignore
ignoredtags = [
        'abbr', 'b', 'br', 'big', 'blockquote', 'center', 'cite', 'em',
        'font', 'h1', 'h2', 'h3', 'h4', 'hiero', 'i', 'kbd', 'nowiki',
        'p', 'plaintext', 's', 'span', 'strike', 'strong',
        'tt', 'u', 'var'
        ]

ignored_tag_patterns = []
for tag in ignoredtags:    
    left = re.compile(r'<%s\b.*?>' % tag, re.IGNORECASE | re.DOTALL)  # both <ref> and <reference>
    right = re.compile(r'</\s*%s>' % tag, re.IGNORECASE)
    ignored_tag_patterns.append((left, right))

# non closing html tags
nonClosingTags = ('br', 'hr', 'nobr')
nonClosing_tag_patterns = [
    re.compile(r'<\s*%s\b[^>]*/?\s*>' % tag, re.DOTALL | re.IGNORECASE) for tag in nonClosingTags
    ]

# self closing html tags
selfClosingTags = ('ref', 'references', 'nowiki')
selfClosing_tag_patterns = [
    re.compile(r'<\s*%s\b[^>]*/\s*>' % tag, re.DOTALL | re.IGNORECASE) for tag in selfClosingTags
    ]

def clean_wiki_text(wikitext):

    # remove section titles since they are otherwise included
    # by compact. Eventually have to refactor compact to avoid this

    # could force pattern to match start and end of lines, but more forceful
    # filtering is not an issue here.
    section_title_pattern = '(=+\s*.+\s*=+)'
    re.sub(section_title_pattern, '', wikitext)

    return compact(clean(wiki2text(transform(wikitext))))

def transform(wikitext):
    """
    Transforms wiki markup.
    @see https://www.mediawiki.org/wiki/Help:Formatting
    """
    # look for matching <nowiki>...</nowiki>
    res = ''
    cur = 0
    for m in nowiki.finditer(wikitext, cur):
        res += transform1(wikitext[cur:m.start()]) + wikitext[m.start():m.end()]
        cur = m.end()
    # leftover
    res += transform1(wikitext[cur:])
    return res


def transform1(text):
    """Transform text not containing <nowiki>"""
    # Drop transclusions (template, parser functions)
    return dropNested(text, r'{{', r'}}')


def wiki2text(text):
    #
    # final part of internalParse().)
    #
    # $text = $this->doTableStuff( $text );
    # $text = preg_replace( '/(^|\n)-----*/', '\\1<hr />', $text );
    # $text = $this->doDoubleUnderscore( $text );
    # $text = $this->doHeadings( $text );
    # $text = $this->replaceInternalLinks( $text );
    # $text = $this->doAllQuotes( $text );
    # $text = $this->replaceExternalLinks( $text );
    # $text = str_replace( self::MARKER_PREFIX . 'NOPARSE', '', $text );
    # $text = $this->doMagicLinks( $text );
    # $text = $this->formatHeadings( $text, $origText, $isMain );

    # Drop tables
    # first drop residual templates, or else empty parameter |} might look like end of table.
    if not options.keep_tables:
        text = dropNested(text, r'{{', r'}}')
        text = dropNested(text, r'{\|', r'\|}')

    # Handle bold/italic/quote
    text = bold_italic.sub(r'\1', text)
    text = bold.sub(r'\1', text)
    text = italic_quote.sub(r'"\1"', text)
    text = italic.sub(r'"\1"', text)
    text = quote_quote.sub(r'"\1"', text)
    
    # residuals of unbalanced quotes
    text = text.replace("'''", '').replace("''", '"')

    # replace internal links
    text = replaceInternalLinks(text)

    # replace external links
    text = replaceExternalLinks(text)

    # drop MagicWords behavioral switches
    text = magicWordsRE.sub('', text)

    # ############### Process HTML ###############

    # turn into HTML, except for the content of <syntaxhighlight>
    res = ''
    cur = 0
    for m in syntaxhighlight.finditer(text):
        res += unescape(text[cur:m.start()]) + m.group(1)
        cur = m.end()
    text = res + unescape(text[cur:])
    return text


def clean(text):
    """
    Removes irrelevant parts from :param: text.
    """

    # Collect spans
    spans = []
    # Drop HTML comments
    for m in comment.finditer(text):
        spans.append((m.start(), m.end()))

    # Drop self-closing tags
    for pattern in selfClosing_tag_patterns:
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end()))

    # Drop non-closing tags
    for pattern in nonClosing_tag_patterns:
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end()))

    # Drop ignored tags
    for left, right in ignored_tag_patterns:
        for m in left.finditer(text):
            spans.append((m.start(), m.end()))
        for m in right.finditer(text):
            spans.append((m.start(), m.end()))

    # Bulk remove all spans
    text = dropSpans(spans, text)

    # Drop discarded elements
    for tag in options.discardElements:
        text = dropNested(text, r'<\s*%s\b[^>/]*>' % tag, r'<\s*/\s*%s>' % tag)

    if not options.toHTML:
        # Turn into text what is left (&amp;nbsp;) and <syntaxhighlight>
        text = unescape(text)

    # Expand placeholders
    for pattern, placeholder in placeholder_tag_patterns:
        index = 1
        for match in pattern.finditer(text):
            text = text.replace(match.group(), '%s_%d' % (placeholder, index))
            index += 1

    text = text.replace('<<', '«').replace('>>', '»')

    #############################################

    # Cleanup text
    text = text.replace('\t', ' ')
    text = spaces.sub(' ', text)
    text = dots.sub('...', text)
    text = re.sub(' (,:\.\)\]»)', r'\1', text)
    text = re.sub('(\[\(«) ', r'\1', text)
    text = re.sub(r'\n\W+?\n', '\n', text, flags=re.U)  # lines with only punctuations
    text = text.replace(',,', ',').replace(',.', '.')
    if options.keep_tables:
        # the following regular expressions are used to remove the wikiml chartacters around table strucutures
        # yet keep the content. The order here is imporant so we remove certain markup like {| and then
        # then the future html attributes such as 'style'. Finally we drop the remaining '|-' that delimits cells.
        text = re.sub(r'!(?:\s)?style=\"[a-z]+:(?:\d+)%;\"', r'', text)
        text = re.sub(r'!(?:\s)?style="[a-z]+:(?:\d+)%;[a-z]+:(?:#)?(?:[0-9a-z]+)?"', r'', text)
        text = text.replace('|-', '')
        text = text.replace('|', '')
    if options.toHTML:
        text = html.escape(text)
    return text



