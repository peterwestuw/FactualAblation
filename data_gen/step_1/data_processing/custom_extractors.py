import json
from profiling import Profiled


def save_to_disk(output_stream, extractor):
    @Profiled.generator
    def save_to_disk(instance):
        extractor.write_instance(output_stream, instance)
        yield 1
    return save_to_disk

class Extractor():
    def write_instance(self, output_stream, instance_dict):
        pass

class NDJsonExtractor(Extractor):
    def write_instance(self, output_stream, instance_dict):
        json_str = json.dumps(instance_dict, indent=None, sort_keys=False, separators=(',', ': '), ensure_ascii=False)
        output_stream.write(json_str + '\n')

class TsvExtractor(Extractor):
    def __init__(self, field_names):
        self.field_names = field_names

    def write_header(self, output_stream):
        output_stream.write("\t".join(self.field_names))

    def write_instance(self, output_stream, instance_dict):
        values = [str(instance_dict.get(f, "")) for f in self.field_names]
        line = "\t".join(values) + '\n'
        output_stream.write(line)

class StringExtractor(Extractor):
    def write_instance(self, output_stream, instance):
        output_stream.write(instance + "\n")