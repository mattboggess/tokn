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

Coming soon.