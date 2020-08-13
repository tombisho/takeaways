# takeaways

This repository has code that supports our work on classification of takeaway food outlet cuisine type on the basis of its name using deep learning. For example, if the classifier is given "Golden Wok" it would hopefully classify this as "South East & East Asian". For more details, please see our paper. Broadly we follow the steps from Edward J. Ross' work on name classification (https://gist.github.com/EdwardJRoss/86b31848a7951411de56f10f55e9de4e). The code included does the following:

1. Prepares raw data from Just Eat into a format suitable for deep learning (data_preparation.ipynb)
2. Generates a language encoder as a means of performing more learning without the need for labelled data (encoder_creation.ipynb)
3. Provides a naive classifier as a comparison to the deep learning model (naive_classifier.ipynb)
4. Allows validation performance, generate summaries of this performance and test individual names (validation_code.ipynb)
5. Code for running as a batch on a High Performance Computing Cluster (batch_run.py and slurm_submit.takeaway )
6. Convert UK Food Standards Agency (FSA) into a format that can be fed into the model for classification (PHE method.ipynb)

We also provide our results from item 6 above, which gives an estimate of the make up of cuisine types from the UK takeaway market.

## Prerequisites

To use this code, it requires the fast.ai deep learning library v1.
A GPU is likely to be needed if any learning is to be done.

## Outputs

In the outputs folder we present the results from applying the classifier to the UK FSA data (07_07_2020_phe_output.csv), and also the generation of the labels from the Just Eat data (07_07_2020_final.csv and 07_07_2020_final_ff.csv - ff denotes all fast food lumped into one category).
