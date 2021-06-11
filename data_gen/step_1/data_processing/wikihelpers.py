
import re, json, pickle, os.path
from mediawiki_parser.preprocessor import make_parser as make_preprocessor
from mediawiki_parser.text import make_parser as make_text_parser
preprocessor = make_preprocessor({})
parser = make_text_parser()

# parses into an AST
from wikitextparser import parse as wtp_parse

from io import StringIO

def parse_wikimarkup(source):
    preprocessed_text = preprocessor.parse(source)
    output = parser.parse(preprocessed_text.leaves())
    return output.value


def estimate_article_count_from_filename(dump_file):
    m = re.search(r"xml-p(\d+)p(\d+)\.", dump_file)
    if m:
        start, end = m.groups()
        return int(end) - int(start)
    return None
    
def page_iterator(xml_lines, id_filter_set = None):
    import untangle
    def process_lines(lines):
        doc = untangle.parse(StringIO("\n".join(lines)))
        return doc.page

    current_lines = []
    in_page = False
    id = None
    for line in xml_lines:
        if re.match(r"\s*<page>", line):
            in_page = True

        if in_page:
            current_lines.append(line)

        if in_page and id is None:
            m = re.match(r"\s*<id>(\d+)</id>", line)
            if m: id = int(m.group(1))
            
        if re.match(r"\s*</page>", line):
            if id_filter_set is None or id is None or id in id_filter_set:
                page = process_lines(current_lines)
                yield page
            current_lines = []
            in_page = False
            id = None
        
    return

def create_xml_index(xml_file):
    with open(xml_file, "rb") as f:
        results = []
        line_idx = 0
        page_start = False
        id = None
        for line in f:
            pos = f.tell() - len(line)
            line_idx += 1
            line = line.decode("utf-8")
            
            if re.match(r"\s*<page>", line):
                page_start = (line_idx, pos)
                
            if re.match(r"\s*</page>", line):
                if page_start:
                    offset = ArticleOffset(id, *page_start, line_idx, pos)
                    results.append(offset)
                page_start = False
                id = None
            
            if page_start and id is None:
                m = re.match(r"\s*<id>(\d+)</id>", line)
                if m: id = int(m.group(1))
        index = dict((a.id, a) for a in results)
        return index

class ArticleOffset():
    def __init__(self, id, line_start, byte_start, line_end, byte_end):
        self.id = id
        self.line_start = line_start
        self.line_end = line_end
        self.byte_start = byte_start
        self.byte_end = byte_end
        
    def __str__(self):
        return "ArticleOffset:" + json.dumps(self.__dict__)
    
    def __repr__(self): return self.__str__()
    
    
class WikiXmlIndex:
    def __init__(self, xml_filename):
        self.xml_filename = xml_filename
        index_filename = xml_filename + ".index_pickle"
        if(os.path.exists(index_filename)):
            self.index = pickle.load(open(index_filename, "rb"))
        else:
            print("Did not find index file for {} in {}, creating index..".format(xml_filename, index_filename))
            self.index = create_xml_index(xml_filename)
            pickle.dump(self.index, open(index_filename, "wb"))
            
    def get_ids(self):
        return self.index.keys()
    
    def lines_for_id(self, id):
        offset_info = self.index[id]
        n_lines = offset_info.line_end - offset_info.line_start
        lines = []
        with open(self.xml_filename, "rb") as f:
            f.seek(offset_info.byte_start)
            for i, line in enumerate(f):
                lines.append(line.decode("utf-8"))
                if i >= n_lines:
                    break
        return lines
    
    def save_id_to_file(self, id):
        lines = self.lines_for_id(id)
        target_filename = os.path.join(os.path.dirname(self.xml_filename), "a{}.xml".format(id))
        with open(target_filename, "w", encoding="utf-8") as f:
            f.writelines(lines)
            
    def save_all_to_files(self):
        for id in self.get_ids():
            print(id)
            self.save_id_to_file(id)
            
            
