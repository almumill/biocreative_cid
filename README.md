## About
This is a bare-bones BERT-based baseline for the BioCreative V Chemical-Disease Relationships (CDR) Chemical-Induced Diseases (CID) task.

## Basic Use

python3 process_data.py <raw CDR data directory> <processed data output directory>
python3 model_es.py <base file name of finetuned model> <output dir for finetuned model> <path to processed training data> <path to processed validation data>
python3 eval.py <predictions output file name> <path to finetuned model> <test>
