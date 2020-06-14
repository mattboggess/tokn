# TOKN Relation Extraction

The code in this folder is adapted from an open source [pytorch deep learning template](https://github.com/victoresque/pytorch-template). Refer to the documentation provided at the link for more detailed information on the structure of this folder.

Training/testing assumes that all of the relation extraction preprocessing scripts have been run and the data splits are in the ../data/relation_extraction folder. Additionally one needs to run the label/apply_label_fns.py script to create the weakly supervised labels for the training set before training/testing. 

## Folder Structure
  ```
  relation_extraction/
  │
  ├── train.py - script for training models 
  ├── test.py - script for evaluating models 
  ├── predict.py - script for predicting relations using model on new input text and term list
  |
  ├── model/ - models, losses, and metrics
  │   ├── model.py: Contains pytorch model architectures
  │   ├── data_loaders.py: Contains data loader classes for reading in batches of data
  │   ├── metric.py: Contains metric functions used to evaluate the model
  │   ├── loss.py: Contains loss functions used to train the model
  │   └── trainer.py: Contains a trainer class that handles the optmization, model saving, etc. for training
  │
  ├── config.json - Configuration file passed to train.py specifying model hyperparameters 
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  | 
  ├── label/ - Label fns and scripts for weak supervision using the Snorkel framework 
  │   ├── apply_label_fns.py: Main script for applying all the label functions to the data 
  │   ├── label_dev.ipynb: Jupyter notebook for interactively developing label functions 
  │   ├── label_constants.py: Important lists/constanst used by all of the label functions 
  │   ├── taxonomy_label_fns.py: Pattern/Heuristic label fns for taxonomy relations 
  │   ├── meronym_label_fns.py: Pattern/Heuristic label fns for meronym relations 
  │   ├── other_label_fns.py: Pattern/Heuristic label fns for OTHER relation 
  │   └── kb_bio101_label_fns.py: Distant supervision label fns using the Bio101 knowledge base 
  │  
  └── utils - Several small utility functions. 
  ```

## Weak Supervision with Snorkel

We don't have labelled training data for the relation extraction task and thus we use a weak supervision framework proposed by Snorkel to programmatically create labels for our data. All of the functionality for this weak supervision is contained in the label folder.

We define a set of label functions that take in a sentence and term pair instance and output one of the relations we are classifying between. So if we have 20 label functions this will give us 20 labels for each instance. We can then aggregate these instances to produce a single label for each instance. The label functions are defined in taxonomy_label_fns.py, meronym_label_fns.py, other_label_fns.py, and kb_bio1010_label_fns.py. The script apply_label_fns.py will apply all of these label functions to our data and then aggregate the results into a single label so that we can train a model.

## Training

The train.py script is responsible for training new models. It requires a json configuration file in order to run: 

`python train.py -c config.json`

See below for a description of the config file, but this specifies all the hyperparameters and the model architecture to be used. As part of the training process, tensorboard visualization information and a log of the terminal output will be written to the saved/logs folder. To visualize the loss and various metrics during or after training in tensorboard run:

`tensorboard --logdir saved/log` 

As part of the training process, a version of the model will be saved to disk each epoch if its performance on some specified validation metric is the best seen so far. These models get saved as follows:

`saved/models/{config_name}/{run_timestamp}/model_best.pth`

Additionally, a copy of the config file used to specify training will also be stored in this same directory. For testing and predicting, one can specify the path to the model_best.pth file for the model of choice to load in the pre-trained weights.

Besides making changes in the config file, one can also change the data splits to train and validate on by changing the argument to the data loader constructions on line 28 and 29 of train.py.

## Evaluating 

The test.py script is responsible for evaluating trained models. It requires a pre-trained model to evaluate as well as a data split to evaluate on as follows:

`python test.py -r saved/models/{config_name}/{run_timestamp}/model_best.pth -s {data_split}`

Valid data splits currently include debug, train, dev, and test.  

This script will load the model and then run the given split through it generating predictions which will then be compared to the labels for evaluation. The script will output several files to the same directory where the saved model lives:

  1. `saved/models/{config_name}/{run_timestamp}/{data_split}-instance-metrics.txt`: This is a text file with precision, recall, and F1 metrics defined across all sentence term-pair instances.
  2. `saved/models/{config_name}/{run_timestamp}/{data_split}-sentence-predictions.pkl`: This is a copy of the split dataframe with two additional columns denoting the predicted relation by the model and its confidence in that prediction. 
  3. `saved/models/{config_name}/{run_timestamp}/{data_split}-term-pair-predictions.pkl`: This is a reduced version of the predictions where we have aggregated predictions to each unique term pair independent of how many sentences they appeared in. For each unique term pair, it provides the true gold label, the label predicted by the KB, the label predicted by the label functions, and the label predicted by the model along with the model's confidence. Term pairs where all four predicted OTHER are removed.
  4. `saved/models/{config_name}/{run_timestamp}/{data_split}-term-pair-metrics.txt`: This is a text file with precision, recall, and F1 metrics defined across all unique term pairs. Thus for each term pair we take the single most likely positive prediction by the model or label function along with the unique label for that term pair from the gold labels and kb labels. This text file outputs the metrics for the model but also for the KB and the pattern label functions.
  
## Predicting

The predict.py script takes in a sentences file (raw text file or a csv file with a sentence column) and a terms file (newline delimited text file or csv with a term column) and outputs predicted relations for every term pair tagged across all of the sentences. Here is the outline for the call to the predict.py script. 

`python predict.py 
     -r saved/models/{config_name}/{run_date_name}/model_best.pth 
     -i {input_sentences_file} 
     -t {input_terms_file} 
     -o {output_dir}`

For example:

`python predict.py 
     -r saved/models/soft_label/0613_003035/model_best.pth 
     -i example_predictions/openstax_bio2e_section4-2.txt
     -t example_predictions/openstax_bio2e_section4-2_hand_labelled_terms.txt
     -o example_predictions`

This script outputs two files:
  - {input_filename}_{config_name}-{run_timestamp}_sentence_predictions.csv: This csv has a separate row for each sentence and term pair pairing. For each of these rows there is a predicted relation and a confidence score from the model.
  - {input_filename}_{config_name}-{run_timestamp}_term_pair_predictions.csv: This csv has a separate row for each unique term pair and is an aggregated version of the first file (excluding OTHER relations). For each term pair that had a non-OTHER prediction we report that term pair, its predicted relation, and its average confidence score.
  
## Config file format

Config files are in `.json` format:
```javascript
{
  "name": "soft_label",         // training session name (folder name given in saved directory) 
  "n_gpu": 1,                   // number of GPUs to use for training.
  
  "arch": {
    "type": "BertEM",           // name of model architecture to train
    "args": {
    }                
  },
  "data_loader": {
    "type": "RelationDataLoader",         // selecting data loader
    "args":{
      "data_dir": "../data/relation_extraction",             // dataset path
      "batch_size": 16,                   // batch size
      "label_type": "hard_label",         // whether to use hard or soft labels for training
      "shuffle": true,                    // shuffle training data before splitting
      "balance_loss": true,               // whether to use class weights in loss function to counteract class imbalance
      "num_workers": 0,                   // number of cpu processes to be used for data loading
      "max_sent_length": 256              // Maximum number of tokens allowed for a sentence
    }
  },
  "optimizer": {
    "type": "AdamW",
    "args":{
      "lr": 3e-5,                     // learning rate
      "correct_bias": false,
      "weight_decay": 0.00            // (optional) weight decay
    }
  },
  "loss": "hard_label_cross_entropy", // loss function, should match label_type
  "metrics": [
        "accuracy", "micro_f1", "macro_f1", "micro_recall", "macro_recall", "micro_precision", "macro_precision" 
  ],
  "lr_scheduler": {
    "type": "StepLR",                  // learning rate scheduler (currently ignored)
    "args":{
      "step_size": 50,          
      "gamma": 0.1
    }
  },
  "trainer": {
    "epochs": 10,                     // number of training epochs
    "save_dir": "saved/",             // checkpoints are saved in save_dir/models/name
    "save_freq": 1,                   // save checkpoints every save_freq epochs
    "verbosity": 2,                   // 0: quiet, 1: per epoch, 2: full
  
    "monitor": "max val_term_f1"      // mode and metric for model performance monitoring. set 'off' to disable.
    "early_stop": 3	                  // number of epochs to wait before early stop. set 0 to disable.
  
    "tensorboard": true,              // enable tensorboard visualization
  }
}
```


