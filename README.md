# TOKN 

This repository holds initial term and relation extraction models as part of the Textbook Open Knowledge Network (TOKN) joint project between Stanford, Rice, and Openstax. The models in this repository were developed by Matt Boggess and Han Lin Aung under the supervision of Dr. Vinay Chaudhri.

Organization:
  - model:
    - re-pytorch-models: Contains pytorch deep learning models for relation extraction. See README within folder for more details.
    - te-pytorch-models: Contains pytorch deep learning models for term extraction. See README within folder for more details.
  - data_processing: Contains scripts for transforming raw data into preprocessed and labeled data for modeling. See pipeline diagram within folder for more details.
  - notebooks: Miscellaneous exploratory notebooks that do error analysis, data validation, etc.
  - book_import: Shared parsing of online textbooks from OpenStax