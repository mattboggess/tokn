# TOKN Term Extraction

The code in this folder is adapted from an open source [pytorch deep learning template](https://github.com/victoresque/pytorch-template). Refer to the documentation provided at the link for more detailed information on the structure of this folder.

Training/testing assumes that all of the term extraction preprocessing scripts have been run and the data splits are in the ../data/term_extraction folder.

## Folder Structure
  ```
  te-pytorch-models/
  │
  ├── train.py - script for training models 
  ├── test.py - script for evaluating models 
  ├── predict.py - script for predicting terms using model on new input text 
  ├── create_term_finetune.py - creates the data based on human validation to retrain model
  │
  ├── config.json - Configuration file passed to train.py specifying model hyperparameters 
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py: Contains pytorch model architectures
  │   ├── data_loaders.py: Contains data loader classes for reading in batches of data
  │   ├── metric.py: Contains metric functions used to evaluate the model
  │   ├── loss.py: Contains loss functions used to train the model
  │   └── trainer.py: Contains a trainer class that handles the optmization, model saving, etc. for training
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils - Several small utility functions. 
  ```

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

  1. `saved/models/{config_name}/{run_timestamp}/{data_split}-metrics.json`: This is a json file containing the metrics on the provided data   split. The token level metrics are computed over the positive BIES tags for each individual token as this is what the model is directly trained on. The term metrics are calculated on the list of terms successfully extracted from the model (valid S and BIE sequences) by comparing these predicted terms to the list of terms in the data split. This more closely aligns with the ultimate goal of the model. Note that we correct the model by trying to join together S tokens that are inappropriately split when the model should have instead used a BIE sequence.

  2. `saved/models/{config_name}/{run_timestamp}/{data_split}-term-classifications.json`: This file contains a list of false positives, false negatives, and true positives. For the false positives and true positives it also includes a confidence score from the model which is the average predicted probability for that term across the split.
  
  3. `saved/models/{config_name}/{run_timestamp}/{data_split}-sentence-term-predictions.pkl`: This file is a pickled pandas dataframe that is a copy of the data split with two additional columns with a list of the predicted terms in the sentence and the associated predicted probabilities for those terms. This file can be used to inspect the model performance at the individual sentence level.

## Predicting

The predict.py script takes in a text file or csv file with a sentence column and outputs predicted terms for the text in that file. Here is the outline for the call to the predict.py script. 

`python predict.py 
     -r saved/models/{config_name}/{run_date_name}/model_best.pth 
     -i {input_text_file} 
     -o {output_dir}`

For example:
`python predict.py 
     -r saved/models/BertSoftmax/0609_175939/model_best.pth 
     -i example_predictions/openstax_bio2e_section4-2.txt
     -o example_predictions`

This script outputs three files:
  - {input_filename}_{config_name}-{run_timestamp}_annotated_text.txt: This is a version of the input text with <term> </term> tags surrounding predicted terms.
  - {input_filename}_{config_name}-{run_timestamp}_predicted_terms_detailed.json: This is a json file with an entry for each distinct term predicted by the model. Each key is the lemmatized version of a term so terms with the same lemmatized form get collapsed. Each term has four entries:
    - type: This is a classification of whether the term is an entity or event. Currently done based on a combination of rules regarding part of speech (verb -> event, noun -> entity) and special handling for nominalizations (nouns ending in ation... -> event). A separate classification for each occurrence of the term in the input text.
    - text: This is the raw text form of the term at each occurrence. This may differ from the lemmatized form if it shows up as plural, etc.
    - indices: These are the locations of the terms in the input text in tokenized form.
    - pos: These are the Spacy part of speech tags for the term for each occurrence of the term.
  - {input_filename}_{config_name}-{run_timestamp}_predicted_terms_simple.csv:  This is a csv file with the following four columns:
    - term: This is the exact representation of the term as it occurred in the text
    - base_term: This is the lemmatized form of the term (there may be multiple terms with the same lemma form)
    - type: Entity or event classification for the term
    - confidence: Average predicted probability by the model for that term

This script first preprocesses the input text to be compatible with the model, gets predictions from the model, and then postprocesses those predictions by tagging the original text. As with the test script, model predictions are modified to merge singleton predictions into phrases where appropriate. Additionally, the tagging has filters to exclude false positives that aren't nouns at the root of a noun phrase or verbs.

## Config file format

Config files are in `.json` format:
```javascript
{
  "name": "BertSoftmax",        // training session name (folder name given in saved directory) 
  "n_gpu": 1,                   // number of GPUs to use for training.
  
  "arch": {
    "type": "BertNER",       // name of model architecture to train
    "args": {
      "dropout_rate": 0.3,   
      "num_classes": 5
    }                
  },
  "data_loader": {
    "type": "TermNERDataLoader",         // selecting data loader
    "args":{
      "data_dir": "../../data/term_extraction",             // dataset path
      "batch_size": 8,                   // batch size
      "shuffle": true,                   // shuffle training data before splitting
      "tags": ["O", "S", "B", "I", "E"], // NER tags representing the classes to predict
      "balance_loss": true,              // whether to use class weights in loss function to counteract class imbalance
      "num_workers": 0,                  // number of cpu processes to be used for data loading
      "max_sent_length": 256             // Maximum number of tokens allowed for a sentence
    }
  },
  "optimizer": {
    "type": "AdamW",
    "args":{
      "lr": 3e-5,                     // learning rate
      "weight_decay": 0.01            // (optional) weight decay
    }
  },
  "loss": "nll_loss",                  // loss
  "sentence_metrics": [
      "sentence_accuracy", "sentence_f1", "sentence_precision", "sentence_recall"
  ],
  "term_metrics": [
      "term_f1", "term_precision", "term_recall"
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


