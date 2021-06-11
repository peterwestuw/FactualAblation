import json
from time import perf_counter
from datetime import timedelta
from operator import itemgetter
from collections import OrderedDict

class StepStats:
    def __init__(self):
        self.elapsed = 0.0
        self.count_in = 0
        
class Profiled:
    """Helper class to add static profiling of functions and generators"""
    perf_stats = OrderedDict()
    total_count = 0

    @classmethod
    def generator(cls, gen_func):
        """Decorate an instance generator with this to add profiling output"""
        cls.perf_stats[gen_func.__name__] = stats = StepStats()
        def wrapped_gen(instance):
            stats.count_in += 1
            start_time = perf_counter()
            result = yield from gen_func(instance)
            stats.elapsed += perf_counter() - start_time
            return result
        wrapped_gen.__name__ = gen_func.__name__
        return wrapped_gen

    @classmethod
    def summarize(cls, processor_functions):
        step_names = [p.__name__ for p in processor_functions if p.__name__ in cls.perf_stats]
        stats = [cls.perf_stats[name] for name in step_names]
        counts = [s.count_in for s in stats]
        elapsed = [timedelta(seconds=s.elapsed) for s in stats]
        if not len(counts) or counts[0] < 1: return 'No stats available'
        total_in = counts[0]

        summary = []
        summary.append("============================")
        summary.append("=== Funneling statistics ===")
        summary.append("Total input count = {}. Total output count = {}. Yield* = {}%".format(
            total_in, cls.total_count, cls.total_count * 100 / total_in))
        for (step, (in_count, out_count)) in zip(step_names, zip(counts, counts[1:] + [cls.total_count])):
            if in_count == 0: continue
            accepted = (out_count * 100) / in_count
            rejected = 100 - accepted
            line = "- {}: in: {} -> out: {} (accepted {:.1f}%, rejected {:.1f}%)".format(step, in_count, out_count, accepted, rejected)
            summary.append(line)

        # fix elapsed: generators are nested
        elapsed = [e - e_next for (e, e_next) in zip(elapsed, elapsed[1:] + [timedelta()])]

        total_elapsed = sum(elapsed, timedelta(0))
        per_item = total_elapsed / total_in
        summary.append("============================")
        summary.append("== Performance statistics ==")
        summary.append("{} elapsed total / avg {} per item".format(total_elapsed, per_item))
        for (step, elapsed, in_count) in sorted(zip(step_names, elapsed, counts), key=itemgetter(1), reverse=True):
            if in_count == 0: continue
            elapsed_per_item = elapsed.total_seconds() * 1000 / in_count 
            percent_total = 100 * elapsed.total_seconds() / total_elapsed.total_seconds()
            line = "- {}: {:.5f} ms per item / {} total ({:.1f}%)".format(step, elapsed_per_item, elapsed, percent_total)
            summary.append(line)

        summary.append("")
        summary.append(json.dumps(dict((k, v.__dict__) for k,v in cls.perf_stats.items())))

        return "\n".join(summary)


        




