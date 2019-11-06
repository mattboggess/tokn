# TOKN Pytorch Term Extraction Models 

The code in this folder is adapted slightly from the following openly available pytorch template:


## Installation 

Install the required Python libraries:

`pip install -r requirements.txt`

Download the Stanford NLP models to enable the text preprocessing pipeline for prediction:

`import stanfordnlp; stanfordnlp.download('en')`

## Training, Testing, & Predicting

### Train 

`python train.py -c config/{name_of_config}.json`

### Test 

The test script will take in any data split in the correct format and evaluate the model on that
data split by producing metrics as well as a term classification breakdown for error analysis.

`python test.py -r saved/models/{config_name}/{run_date_name}/{model_name}.pth -s {split_type}`

Examples:
 - Test weighted Bert model on life biology test data:
    `python test.py -r saved/models/TE_BertNER/weighted/weighted_model_best.pth -s life_test`
 - Test unweighted Bert model on openstax biology validation data:
    `python test.py -r saved/models/TE_BertNER/unweighted/unweighted_model_best.pth -s validation`

Outputs two files to the same folder defined by r argument:
  - {split}-{model}-classifications.json: json file with terms split across false positive, 
    false negatives, and true positives
  - {split}-{model}-metrics.json: json file with metrics for the evaluation. sentence metrics
    count across all sentences (with repetitions for terms). term metrics take all terms classified
    at least once and calculate metrics with this list against the list of tagged terms

### Predict

The predict.py script takes in a text file and outputs predicted terms for that the text in that file.

`python predict.py -r saved/models/{config_name}/{run_date_name}/{model_name}.pth 
  -i {input_text_file} -o {output_dir}`

Example:
   
 `python predict.py -r saved/models/TE_BertNER/weighted/weighted_model_best.pth -i example_passage.txt`

## Folder Structure
  ```
  te-pytorch-models/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  ├── predict.py - predict set of terms from input text 
  │
  ├── config - folder that holds json configuration files 
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py: Contains pytorch model architectures
  │   ├── data_loaders.py: Contains data loader classes for reading in batches of data
  │   ├── metric.py: Contains metric functions used to evaluate the model
  │   ├── loss.py: Contains loss functions used to train the model
  │   └── trainer.py: Contains a trainer class used to train the model
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
  └── utils/ - small utility functions
      ├── util.py
  ```

## Config file format

Config files are in `.json` format:
```javascript
{
  "name": "Mnist_LeNet",        // training session name (should match json file name)
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
      "batch_size": 64,                // batch size
      "shuffle": true,                 // shuffle training data before splitting
      "num_workers": 2,                // number of cpu processes to be used for data loading
      "embedding_type": "Bert",         // Type of embedding to use
    }
  },
  "optimizer": {
    "type": "Adam",
    "args":{
      "lr": 0.001,                     // learning rate
      "weight_decay": 0,               // (optional) weight decay
      "amsgrad": true
    }
  },
  "loss": "nll_loss",                  // loss
  "metrics": [
    "accuracy", "top_k_acc"            // list of metrics to evaluate
  ],                         
  "lr_scheduler": {
    "type": "StepLR",                  // learning rate scheduler
    "args":{
      "step_size": 50,          
      "gamma": 0.1
    }
  },
  "trainer": {
    "epochs": 100,                     // number of training epochs
    "save_dir": "saved/",              // checkpoints are saved in save_dir/models/name
    "save_freq": 1,                    // save checkpoints every save_freq epochs
    "verbosity": 2,                    // 0: quiet, 1: per epoch, 2: full
  
    "monitor": "min val_loss"          // mode and metric for model performance monitoring. set 'off' to disable.
    "early_stop": 10	                 // number of epochs to wait before early stop. set 0 to disable.
  
    "tensorboard": true,               // enable tensorboard visualization
  }
}
```

