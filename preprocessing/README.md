# Data Preprocessing Pipeline

This folder holds scripts that preprocess the textbooks and knowledge base used as training/evaluation
data for term and relation extraction. All scripts should be run in the order described here to 
ensure all dependencies for each script are met.

## Extract Sentences from Raw Book Formats

These scripts turn raw HTML/cnxml versions of the books into separate cleaned sentences:

  - parse_life_bio_sentences.py: Extracts sentences from html version of Life Biology textbook
  - openstax_book_import: This folder holds scripts that extract sentences from cnxml versions of
    multiple OpenStax textbooks. The parse_sentences.py script must be run with each Openstax book specified in
    the book_title variable.

## Collect Term List and Preprocessed Sentences

These scripts create a full comprehensive term list and clean textbooks sentences that
have both been preprocessed with Spacy:

  - preprocess_textbooks.py: Extracts key terms from the textbooks parse into separate text files
    and filters textbooks sentences to relevant sentences and saves them out with spacy preprocessing
  - preprocess_kb_bio101_terms.py: Pulls out terms from the Bio101 knowledge base into a separate text file
  - Manually curated hand labelled term lists should be manually copied to the data/preprocessed/terms folder at this point 
    before running the next script. 
  - collect_terms.py: Assembles all terms from all the different sources into a single dataframe and
    Spacy preprocesses the terms. Previous two scripts must be run first and any hand-labelled term lists
    need to be manually copied in to the data/preprocessed/terms folder.

## Create Term Extraction Data

These scripts result in a train/dev/test split for term extraction modelling:

  - tag_sentences.py: This tags every term in each sentence for a set of textbooks. The result is
    a set of sentences each with a set of tagged terms for each sentence. Set tag_type = 
    "term_extraction" to use the set of textbooks/terms for term extraction.
  - split_term_extraction_data.py: This takes the tagged sentences and assembles them into 
    train, dev, and test sets ensuring there is no overlap between the train and dev/test sets.

## Create Relation Extraction Data

These scripts result in a train/dev/test split for relation extraction modelling:

  - prep_life_bio_kb.py: This creates lookup dictionaries that can be used to check if a term pair
    has a relation in the Bio101 KB or not. It is used for generating training labels for
    relation extraction.
  - tag_sentences.py: This tags every term in each sentence for a set of textbooks. The result is
    a set of sentences each with a set of tagged terms for each sentence. Set tag_type = 
    "relation_extraction" to use the set of textbooks/terms for relation extraction.
  - generate_term_pairs: This script takes the tagged sentences and enumerates all term pairs
    in the sentences to create a new dataframe with one row for every sentence/term pair pairing.
  - split_relation_extraction_data.py: This takes the sentence/term pairs and assembles them into 
    train, dev, and test sets ensuring there is no overlap between the train and dev/test sets.

## Helper Scripts

  - data_processing_utils.py: This contains important data processing utility functions used by the above
    scripts. The most important of which is the term tagging function that has a bunch of functionality
    for ensuring proper tagging of terms within sentences.
  - test_data_processing_utils.py: Tests for the utils functions.