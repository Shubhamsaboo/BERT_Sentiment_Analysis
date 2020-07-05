# BERT based Sentiment Analysis

## Dataset Used

IMDB movie review dataset consisting of 50K movie reviews with the corresponding sentiments labeled as "Positive" and "Negative".

This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. The dataset consists of 25,000 highly polar movie reviews for training and 25,000 for testing. For more dataset information, please go through the following link: http://ai.stanford.edu/~amaas/data/sentiment/

## Models Used
BERT Base Uncased English (bert_en_uncased_L-12_H-768_A-12)
https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2

### BERT Overview
BERT (Bidirectional Encoder Representations from Transformers) provides dense vector representations for natural language by using a deep, pre-trained neural network with the Transformer architecture. This TF Hub model uses the implementation of BERT from the TensorFlow Models repository on GitHub at tensorflow/models/official/nlp/bert. It uses L=12 hidden layers (i.e., Transformer blocks), a hidden size of H=768, and A=12 attention heads.

This model has been pre-trained for English on the Wikipedia and BooksCorpus using the code published on GitHub. Inputs have been "uncased", meaning that the text has been lower-cased before tokenization into word pieces, and any accent markers have been stripped. For training, random input masking has been applied independently to word pieces (as in the original BERT paper).

### Train/Validation Composition
Training Data - 90% of 50K movie reviews (45k)
Validation Data - 10% of 50K movie reviews (5k)

### Metrics Used 
Loss Function - Binary Crossentropy
