# Splits relation extraction data into train and test splits for model evaluation 
import json
import numpy as np

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
    null_word_pairs = list(relations["no-relation"].keys())
    
    for relation in relations:
        if relation != "no-relation":
            train_relations[relation] = [] 
            test_relations[relation] = [] 
            validation_relations[relation] = []
            mini_relations[relation] = []
            
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
            
            for split in ["train", "validation", "test", "mini"]:
                if split == "train":
                    db = train_relations
                    pos_word_pairs = train_sample
                elif split == "validation":
                    db = validation_relations
                    pos_word_pairs = validation_sample
                elif split == "test":
                    db = test_relations
                    pos_word_pairs = test_sample
                else:
                    db = mini_relations
                    pos_word_pairs = test_sample[:min(5, len(test_sample))]
            
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
    
    with open(f"{data_dir}/relations_train.json", "w") as f:
        json.dump(train_relations, f, indent=4)
    with open(f"{data_dir}/relations_validation.json", "w") as f:
        json.dump(validation_relations, f, indent=4)
    with open(f"{data_dir}/relations_test.json", "w") as f:
        json.dump(test_relations, f, indent=4)
    with open(f"{data_dir}/relations_debug.json", "w") as f:
        json.dump(mini_relations, f, indent=4)
                
    
    
            
    
    