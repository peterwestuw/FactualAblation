import json
    
def get_FA_wiki_dataset():
    dataset_cf_wiki = []
    with open('data/FA_wiki_examples.jsonl') as f:
        for line in f:
            ex = json.loads(line)

            if not ('|' in ex['target'] or len(ex['target']) < 50 or (not ex['target'].endswith('.'))):
                dataset_cf_wiki += [(ex['grounding_true'], ex['grounding_false'], ex['context'], ex['target'])]

    dataset_cf_wiki = list(set(dataset_cf_wiki))
    print(len(dataset_cf_wiki))
    
    return dataset_cf_wiki
       
    
def get_FA_synthetic_dataset():
    dataset_cf_constructed = []

    with open('data/FA_synthetic_examples.jsonl') as f:
        for line in f:
            d = json.loads(line)

            for i, ex in enumerate(d.values()):
                # this example is broken
                if i != 14:
                    dataset_cf_constructed += [(ex['source'], ex['new_source'], ex['context'], ex['target'])]
                    dataset_cf_constructed += [(ex['new_source'], ex['source'], ex['context'], ex['new_target'])]
    return dataset_cf_constructed
