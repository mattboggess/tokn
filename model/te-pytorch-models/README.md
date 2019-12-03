# TOKN Pytorch Term Extraction Models 

The code in this folder is adapted from an open source [pytorch deep learning template](https://github.com/victoresque/pytorch-template). Refer to
the documentation provided at the link for more detailed information on the structure of this repo.

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
  └── utils/ - Contains important data preprocessing functions for the predict script
  ```

## Installation & Environment

### Local

Install the required Python libraries:

`pip install -r requirements.txt`

Download the Stanford NLP models to enable the text preprocessing pipeline for prediction:

`import stanfordnlp; stanfordnlp.download('en')`

### AWS 

Training was done on a virtual AWS machine with GPU. We used the [AWS Deep Learning AMI (Ubuntu)](https://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/) image on a p2.xlarge instance with a single GPU. The image comes with a bunch of different pre-configured conda environments. We used the pytorch_p36 environment (conda activate pytorch_p36) and then pip installed additional Python packages as needed.

## Training

The train.py script is responsible for training new models. It requires a json configuration file in order to run: 

`python train.py -c config/{name_of_config}.json`

See below for a description of the config file, but this specifies all the hyperparameters and the model architecture to be used. As part of the training process, tensorboard visualization information and a log of the terminal output will be written to the saved/logs folder. To visualize the loss and various metrics during or after training in tensorboard run:

`tensorboard --logdir saved/log` 

As part of the training process, a version of the model will be saved to disk each epoch if its performance on some validation metric is the best seen so far. These models get saved as follows:

`saved/models/{config_name}/{run_timestamp}/model_best.pth`

Additionally, a copy of the config file used to specify training will also be stored in this same directory. For testing and predicting, one can specify the path to the model_best.pth file for the model of choice to load in the pre-trained weights.

Besides making changes in the config file, one can also change the data splits to train and validate on by changing the argument to the data loader constructions on line 28 and 29 of train.py.

## Evaluating 

The test.py script is responsible for evaluating trained models. It requires a pre-trained model to evaluate as well as a data split to evaluate on as follows:

`python test.py -r saved/models/{config_name}/{run_timestamp}/model_best.pth -s {data_split}`

Valid data splits currently include debug (small data set for debugging purposes), train (all openstax STEM textbooks excluding biology), validation (openstax Biology), psych_test (openstax Psychology), and life_test (selected sentences from Life Biology textbook).

This script will load the model and then run the given split through it generating predictions which will then be compared to the labels for evaluation. The script will output two files to the same directory where the saved model lives:

`saved/models/{config_name}/{run_timestamp}/{data_split}-model_best-metrics.json`

This is a json file containing the metrics on the provided data split. The metrics are common classification metrics defined at two levels:

1. Sentence-level metrics: The models provided here all make term predictions at the sentence level. This means that each term needs to be predicted as many times as it appears in the text. The sentence level metrics look at how many times the model correctly predicts terms across all appearances and is thus more sensitive to terms that appear more often.
2. Term-level metrics: For these metrics, all terms predicted in at least one sentence by the model for the given input data are considered terms discovered by the model. The term level metrics thus compare this extracted list of terms by the model to the overall list of terms given in the input data. This weights each term equally.

`saved/models/{config_name}/{run_timestamp}/{data_split}-model_best-term-classifications.json`

This second file contains false_positives, true_positives, and false_negatives terms comparing the model's output to the provided list of terms.

## Predicting

The predict.py script takes in a text file and outputs predicted terms for the text in that file. Here is the outline for the call to the predict.py script. 

`python predict.py 
     -r saved/models/{config_name}/{run_date_name}/{model_name}.pth 
     -i {input_text_file} 
     -o {output_dir}`

For example:
`python predict.py 
     -r saved/models/BertSoftmax/1129_025337/model_best.pth 
     -i example_predictions/openstax_bio_ch3.txt
     -o example_predictions`

This script outputs two files:
  - {input_filename}_{config_name}-{run_timestamp}_annotated_text.txt: This is a version of the
    input text with <entity> </entity> and <event> </event> tags surrounding terms predicted as entities and events specifically.
  - {input_filename}_{config_name}-{run_timestamp}_predicted_terms.json: This is a json file with an entry for each distinct term predicted by the model. Each key is the lemmatized version of a term so terms with the same lemmatized form get collapsed. Each term has four entries:
    - type: This is a classification of whether the term is an entity or event. Currently done based on a combination of rules regarding part of speech (verb -> event, noun -> entity) and special handling for nominalizations (nouns ending in ation... -> event). A separate classification for each occurrence of the term in the input text.
    - text: This is the raw text form of the term at each occurrence. This may differ from the lemmatized form if it shows up as plural, etc.
    - indices: These are the locations of the terms in the input text in tokenized form.
    - pos: These are the Spacy part of speech tags for the term for each occurrence of the term.

This script first preprocesses the input text to be compatible with the model, gets predictions from the model, and then postprocesses those predictions. The postprocessing includes merging of consecutive predicted singleton terms into a single term phrase where appropriate since the model has a bias towards predicting the words in phrases individually rather than as a full phrase. Additionally, it has a filter to exclude false positives that aren't nouns, adjectives, or verbs. These merged and filtered terms are then used to tag the input text resulting in the two files described above.

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
      "num_workers": 0,                  // number of cpu processes to be used for data loading
      "embedding_type": "Bert",          // Type of embedding to use
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
    "type": "StepLR",                  // learning rate scheduler
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


