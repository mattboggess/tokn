# Snorkel Based Relation Extraction

# preprocessing

This folder contains preprorcessing scripts that transform parsed sentence files for OpenStax Biology and Life Biology (csv/text files with one sentence per line) into pandas DataFrames with one sentence/term pair grouping per row via the following steps:

  1. collect_terms.py: Pulls in terms from OpenStax Biology glossary and Life Biology KB to make a candidate list of terms to tag sentences.
  2. tag_sentences.py: Tags every term pair in each sentence from OpenStax Biology and Life Biology using the terms output from step 1 and assembles each sentence/term pair grouping into a separate row in a data frame. The key workhorse for this is the tag_terms function found in data_processing_utils.py.
  3. split_data.py: Takes the sentences with term pairs tagged from step 2 and splits them into a train/dev/test set along with a few other minor cleanups.
  
# label

This folder contains all of the Snorkel label functionality:

- taxonomy_label_fns.py: This file contains all of the snorkel label functions created for taxonomy relations. They are a mixture of text heuristics, Spacy dependency parse pattern matches, and distant supervision from the KB101 knowledge base. See the accompanying test file for examples of inputs and expected outputs.
- apply_label_fns.py: This script applies all of the label functions from taxonomy_label_fns.py to each of the three data splits creating a labelled DataFrame for each split that has additional columns added with the label from each label function. Also creates hard labels via majority vote across label functions and soft labels via Snorkel's LabelModel that can be used to train models.
- label_dev.ipynb: Notebook used to explore ideas for labelling functions/troubleshoot existing labelling functions. Also includes example of how to visualize Spacy dependency parses.

# model_prep.ipynb

Last step in prepping the data for modelling. This notebook takes in the labelled DataFrames and does a few additional tweaks to make them ready for modelling (standardizing the hard and soft label columns, subsamples the train data negative examples, creates a debug set, etc.). See the notebook for more details. It also provides a lot of counts/plots of the data such as class imbalance.

# models

The code in this folder is adapted from an open source [pytorch deep learning template](https://github.com/victoresque/pytorch-template). Refer to the documentation provided at the link for more detailed information on the structure of this repo.

- train.py & config.json: This is the main training script to train a model. It gets invoked with the config file that holds a bunch of parameter settings to use while training (such as learning rate, whether to balance loss function, etc.). One needs to make sure the correct splits are passed to the training data loader and the validation data loader and the correct config parameters are set (a copy of the config file gets saved to the model folder). Then you can train with the following:

	python train.py -c config.json

- test.py: This is the script for evaluating a model on any given split. It requires a saved model and then will produce predictions for the entire split in the form of a pickled data frame along with term level error classifications by relation (false positives, false negatives, etc.). I t also computes precision, recall, and f1 for each of these. These outputs are saved in the model folder (all models are saved in the saved folder). Example:

	python test.py -r saved/models/BertEM/0508_132777/model_best.pth -s dev

- model: This folder contains the main files that have been adapted from the template:

  - data_loaders.py: This holds the pytorch data loader that is responsible for converting the Pandas DataFrame input into batched Bert-compatible representations.
  - model.py: This holds the code for the Pytorch models. Currently only model is the BertEM model.
  - metric.py: This holds functions for computing metrics such as f1, etc.
  - loss.py: This holds loss functions for the models
  - trainer.py: This holds the main class that does the training. The train_epoch and valid_epoch methods are the key places that have been edited.
