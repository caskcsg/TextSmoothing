Source code for ACL 2022 paper: Text Smoothing: Enhance Various Data Augmentation Methods on Text Classification Tasks

Our work mainly based on [Data Augmentation using Pre-trained Transformer Models](https://github.com/amazon-research/transformers-data-augmentation)

Code contains implementation of the following data augmentation methods
- TextSmoothing
- EDA + TextSmoothing
- Backtranslation + TextSmoothing
- CBERT + TextSmoothing
- BERT Prepend + TextSmoothing
- GPT-2 Prepend + TextSmoothing
- BART Prepend + TextSmoothing

## DataSets 

In paper, we use three datasets from following resources
 - STSA-2 : [https://github.com/1024er/cbert_aug/tree/crayon/datasets/stsa.binary](https://github.com/1024er/cbert_aug/tree/crayon/datasets/stsa.binary)
 - TREC : [https://github.com/1024er/cbert_aug/tree/crayon/datasets/TREC](https://github.com/1024er/cbert_aug/tree/crayon/datasets/TREC)
 - SNIPS : [https://github.com/MiuLab/SlotGated-SLU/tree/master/data/snips](https://github.com/MiuLab/SlotGated-SLU/tree/master/data/snips)

### Low-data regime experiment setup
Run `src/utils/download_and_prepare_datasets.sh` file to prepare all datsets.
`download_and_prepare_datasets.sh` performs following steps
1. Download data from github 
2. Replace numeric labels with text for STSA-2 and TREC dataset
3. For a given dataset, creates 15 random splits of train and dev data.

## Dependencies 
 
To run this code, you need following dependencies 
- Pytorch 1.5
- fairseq 0.9 
- transformers 2.9 

## How to run 
To run data augmentation experiment for a given dataset, run bash script in `scripts` folder.
For example, to run data augmentation on `snips` dataset, 
 - run `scripts/bart_snips_lower.sh`  for BART experiment 
 - run `scripts/bert_snips_lower.sh` for rest of the data augmentation methods 


   


