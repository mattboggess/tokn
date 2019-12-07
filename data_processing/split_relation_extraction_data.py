# Splits relation extraction data into train and test splits for model evaluation 
import json
import numpy as np
import re

TEST_FRACTION = 0.15
VALIDATION_FRACTION = 0.15

if __name__ == "__main__":
    
    np.random.seed(2020)
    
    data_dir = "../data/relation_extraction"
    
    with open(f"{data_dir}/relations_db.json", "r") as f:
        relations = json.load(f)
    
    # remove relations with no data
    for relation in relations:
        relations[relation] = {k: v for k, v in relations[relation].items() 
                               if len(relations[relation][k]["sentences"]) > 0}
    
    
    train_relations = {}
    validation_relations = {}
    test_relations = {} 
    mini_relations = {}
    full_relations = {}
    null_word_pairs = list(relations["no-relation"].keys())
    
    for relation in relations:
        if relation != "no-relation":
            train_relations[relation] = [] 
            test_relations[relation] = [] 
            validation_relations[relation] = []
            mini_relations[relation] = []
            full_relations[relation] = []
            
            # split the data
            word_pairs = list(relations[relation].keys())
            count_word_pairs = len(word_pairs)
            
            # extract train samples
            train_sample = np.random.choice(
                word_pairs, 
                int(count_word_pairs * (1 - TEST_FRACTION - VALIDATION_FRACTION)),
                replace=False)
            eval_sample = [wp for wp in word_pairs if wp not in train_sample]
            
            # split the rest into validation and test
            count_word_pairs = len(eval_sample)
            validation_sample = np.random.choice(
                eval_sample, 
                int(count_word_pairs * (1 - VALIDATION_FRACTION / (TEST_FRACTION + VALIDATION_FRACTION))),
                replace=False)
            test_sample = [wp for wp in eval_sample if wp not in validation_sample]
            
            for split in ["train", "validation", "test", "debug", "full"]:
                if split == "train":
                    db = train_relations
                    pos_word_pairs = train_sample
                elif split == "validation":
                    db = validation_relations
                    pos_word_pairs = validation_sample
                elif split == "test":
                    db = test_relations
                    pos_word_pairs = test_sample
                elif split == "debug":
                    db = mini_relations
                    pos_word_pairs = test_sample[:min(5, len(test_sample))]
                else:
                    db = full_relations
                    pos_word_pairs = word_pairs 
            
                # match an equal number of negative word pairs
                neg_word_pairs = np.random.choice(null_word_pairs, 
                                                  len(pos_word_pairs),
                                                  replace=False)
                null_word_pairs = [wp for wp in null_word_pairs if wp not in neg_word_pairs]
            
                # add negative and positive word pair bags
                for pos_word_pair, neg_word_pair in zip(pos_word_pairs, neg_word_pairs):

                    pos_examples = relations[relation][pos_word_pair]["sentences"]
                    neg_examples = relations["no-relation"][neg_word_pair]["sentences"]
                    db[relation].append({pos_word_pair: {"sentences": pos_examples, 
                                                         "relation": relation}})
                    db[relation].append({neg_word_pair: {"sentences": neg_examples, 
                                                         "relation": "no-relation"}})
                    
                    # add no-relation flipped version of word pair so it learns directionality
                    rep = {"e1": "e2", "e2": "e1"}
                    pattern = re.compile("|".join(rep.keys()))
                    #flip_wp = " -> ".join(pos_word_pair.split(" -> ")[::-1])
                    #flip_examples = [pattern.sub(lambda m: rep[re.escape(m.group(0))], pe) for pe in pos_examples] 
                    #db[relation].append({flip_wp: {"sentences": flip_examples, 
                    #                               "relation": "no-relation"}})
    
    with open(f"{data_dir}/relations_train.json", "w") as f:
        json.dump(train_relations, f, indent=4)
    with open(f"{data_dir}/relations_validation.json", "w") as f:
        json.dump(validation_relations, f, indent=4)
    with open(f"{data_dir}/relations_test.json", "w") as f:
        json.dump(test_relations, f, indent=4)
    with open(f"{data_dir}/relations_debug.json", "w") as f:
        json.dump(mini_relations, f, indent=4)
    with open(f"{data_dir}/relations_full.json", "w") as f:
        json.dump(full_relations, f, indent=4)
                
    
    
            
    
    