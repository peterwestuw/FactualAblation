from functools import reduce

def chain_generators(iterable, list_of_generators):
    """
    Takes a base iterable and a list of generators g_n(x).
    Yields a generator that produces the flattened sequence from function-chaining iterators g_n(g_n-1(..(g_1(x)))) 
    """

    def flatten1(nested_iterable):
        return (item for inner in nested_iterable for item in inner)

    # IMPORTANT: this function can NOT be inlined. Python scope/shadowing intricacies would break it
    def apply(gen, iterable): 
        return (gen(x) for x in iterable)

    def flatmap(iter, gen):
        return flatten1(apply(gen, iter))

    return reduce(flatmap, list_of_generators, iterable)

def process(input_stream, base_generator, processors):
    """Applies the base_generator on input_stream, then chains processor steps in processors, finally uses extractor to write to output_stream"""
    from tqdm import tqdm
    from profiling import Profiled

    iterable = tqdm(base_generator(input_stream), "baseline generator", mininterval=3.0)
    results = chain_generators(iterable, processors)
    for _ in tqdm(results, "final results", mininterval=3.0):
        Profiled.total_count += 1

if __name__ == "__main__":
    """This is a demo of how generators are chained"""

    def gen_numbers(n, m):
        for i in range(n, m):
            yield i

    def gen_even(number):
        if number % 2 == 0:
            yield number
        return

    def gen_digits(number):
        for c in str(number):
            yield c

    final_generator = chain_generators(
        gen_numbers(10, 20),
        [
            gen_even,
            gen_digits
        ])

    print(list(final_generator))
