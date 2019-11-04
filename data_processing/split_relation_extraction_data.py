# Splits relation extraction data into train and test splits for model evaluation 
import json
import numpy as np

TEST_FRACTION = 0.1

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
    test_relations = {} 
    mini_relations = {}
    null_word_pairs = list(relations["no-relation"].keys())
    
    for relation in relations:
        if relation != "no-relation":
            train_relations[relation] = [] 
            test_relations[relation] = [] 
            mini_relations[relation] = []
            
            word_pairs = list(relations[relation].keys())
            count_word_pairs = len(word_pairs)
            train_sample = np.random.choice(word_pairs, 
                                            int(count_word_pairs * (1 - TEST_FRACTION)),
                                            replace=False)
            test_sample = [wp for wp in word_pairs if wp not in train_sample]
            
            for split in ["train", "test", "mini"]:
                if split == "train":
                    db = train_relations
                    pos_wps = train_sample
                elif split == "test":
                    db = test_relations
                    pos_wps = test_sample
                else:
                    db = mini_relations
                    pos_wps = test_sample[:min(5, len(test_sample))]
            
                neg_wps = np.random.choice(null_word_pairs, 
                                           len(pos_wps),
                                           replace=False)
            
                for pos_pair, neg_pair in zip(pos_wps, neg_wps):

                    pos_examples = relations[relation][pos_pair]["sentences"]
                    neg_examples = relations["no-relation"][neg_pair]["sentences"]
                    db[relation].append({pos_pair: {"sentences": pos_examples, 
                                                    "relation": relation}})
                    db[relation].append({neg_pair: {"sentences": neg_examples, 
                                                    "relation": "no-relation"}})
    
    with open(f"{data_dir}/relations_train.json", "w") as f:
        json.dump(train_relations, f)
    with open(f"{data_dir}/relations_test.json", "w") as f:
        json.dump(test_relations, f)
    with open(f"{data_dir}/relations_debug.json", "w") as f:
        json.dump(mini_relations, f)
                
    
    
            
    
    