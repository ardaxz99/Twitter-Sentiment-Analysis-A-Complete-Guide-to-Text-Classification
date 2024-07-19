# Twitter Sentiment Analysis - A Complete Guide to Text Classification

This repository presents a comprehensive guide to performing sentiment analysis on a large dataset of tweets. The project involves classifying the sentiment of tweets as positive or negative, providing insights into public opinion on various topics.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Goals](#goals)
- [Methods](#methods)
  - [TF-IDF Vectorization with Unigram](#tf-idf-vectorization-with-unigram)
  - [TF-IDF Vectorization with N-grams](#tf-idf-vectorization-with-n-grams)
  - [Word2Vec Trained from Scratch](#word2vec-trained-from-scratch)
  - [Doc2Vec Trained from Scratch](#doc2vec-trained-from-scratch)
  - [Google News Word2Vec](#google-news-word2vec)
  - [Glove Vectorization](#glove-vectorization)
  - [Gensim Fasttext Trained from Scratch](#gensim-fasttext-trained-from-scratch)
  - [BERT](#bert)
  - [RoBERTa](#roberta)
  - [Latent Dirichlet Allocation (LDA)](#latent-dirichlet-allocation-lda)
  - [Universal Sentence Encoder](#universal-sentence-encoder)
  - [Sentence Transformers](#sentence-transformers)
  - [ELMo](#elmo)
  - [CLIP](#clip)
- [Results](#results)
- [Future Work](#future-work)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)
- [License](#license)

## Introduction

Sentiment analysis involves classifying text data to determine whether the sentiment is positive or negative. In this project, we analyze a large dataset of tweets to uncover public opinion on various topics.

## Dataset

The dataset used for this project is the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140), which contains 1.6 million tweets labeled with sentiment: 0 for negative and 4 for positive. The dataset includes the following columns:
- `target`: Sentiment label (0 = negative, 4 = positive)
- `ids`: Tweet ID
- `date`: Date of the tweet
- `flag`: Query flag
- `user`: Username
- `text`: Tweet content

## Goals

1. To clean and preprocess the tweet data.
2. To explore the data through various visualizations and descriptive statistics.
3. To build and evaluate machine learning and deep learning models for sentiment classification.

## Methods

We employed various methods to extract feature embeddings from the tweet text, which were then used as input for classification models. Below are the feature extraction methods:

### TF-IDF Vectorization with Unigram

TF-IDF (Term Frequency-Inverse Document Frequency) evaluates the importance of a word in a document relative to the corpus. It converts text data into numerical features for machine learning algorithms by capturing the significance of words based on their frequency and rarity in the corpus.

### TF-IDF Vectorization with N-grams

Extending TF-IDF to N-grams captures more contextual information by considering combinations of words. This approach enhances the model's understanding of word dependencies and phrases, providing richer features for text data.

### Word2Vec Trained from Scratch

Word2Vec converts text data into numerical vectors by capturing semantic relationships between words. It uses either the Continuous Bag of Words (CBOW) or Skip-gram models to predict the context of words in a sentence, creating dense word vectors.

### Doc2Vec Trained from Scratch

Doc2Vec extends Word2Vec to generate vector representations of entire documents. It uses Distributed Memory (DM) and Distributed Bag of Words (DBOW) models to capture semantic relationships between sentences and paragraphs.

### Google News Word2Vec

The Google News Word2Vec model is a pre-trained Word2Vec model trained on a large corpus of news articles. It provides high-quality embeddings that capture a wide range of semantic relationships and contextual understandings.

### Glove Vectorization

GloVe (Global Vectors for Word Representation) is a pre-trained word embedding model that uses aggregated global word-word co-occurrence statistics. It captures semantic relationships between words by leveraging the overall statistical information of a corpus.

### Gensim Fasttext Trained from Scratch

FastText extends Word2Vec by representing words as a bag of character n-grams, capturing sub-word information. This approach addresses out-of-vocabulary (OOV) words and generates more accurate word vectors.

### BERT

BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model that reads entire sequences of words simultaneously. This bidirectional approach captures the context of words based on their surroundings, improving the understanding of meaning.

### RoBERTa

RoBERTa (Robustly Optimized BERT Pretraining Approach) is an improved version of BERT with larger training data, longer training periods, and dynamic masking. It removes the Next Sentence Prediction (NSP) task and focuses on masked language modeling.

### Latent Dirichlet Allocation (LDA)

LDA is a generative probabilistic model used for topic modeling. It discovers underlying topics in a collection of documents by representing each document as a mixture of topics and each topic as a distribution over words.

### Universal Sentence Encoder

The Universal Sentence Encoder (USE) encodes text into fixed-length vectors that capture semantic meaning. It uses a transformer-based architecture to generate embeddings that excel at capturing contextual information of sentences.

### Sentence Transformers

Sentence Transformers generate dense vector representations for sentences, capturing their semantic meaning. They extend traditional transformers by fine-tuning them for generating meaningful sentence embeddings.

### ELMo

ELMo (Embeddings from Language Models) generates context-dependent word embeddings using a bi-directional LSTM. It captures the meaning of words based on their context within a sentence, providing more accurate word representations.

### CLIP

CLIP (Contrastive Language–Image Pre-training) bridges the gap between vision and language by learning visual concepts from text. It uses large-scale natural language supervision to learn joint embeddings for images and their descriptions.

## Classifiers

For sentiment classification, we used a variety of classical machine learning models as well as a single hidden layer neural network. Below are the classifiers employed in this project:

### Classical Machine Learning Models

- `CatBoostClassifier`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `ExtraTreesClassifier`
- `BaggingClassifier`
- `AdaBoostClassifier`
- `GradientBoostingClassifier`
- `KNeighborsClassifier`
- `LogisticRegression`
- `SGDClassifier`
- `XGBClassifier`
- `LinearSVC`

### Neural Network

In addition to the classical machine learning models, we also implemented a single hidden layer neural network using PyTorch. The neural network architecture consists of:
- **Input Layer**: Takes in the TF-IDF features.
- **Hidden Layer**: Contains 128 neurons with ReLU activation.
- **Output Layer**: A single neuron with Sigmoid activation for binary classification.

The neural network is trained using Binary Cross-Entropy Loss and the Adam optimizer over 300 epochs.

## Results

The performance of our sentiment analysis models was evaluated using various metrics. The results demonstrate the effectiveness of different methods in accurately predicting the sentiment of tweets.

## Future Work

This project has covered a wide range of methods for sentiment analysis on Twitter data. However, there are numerous avenues for further exploration and enhancement. Below are some potential directions for future work:

### Feature Extraction and Text Embedding

1. **Transformers-based Models**
   - **DistilBERT**: Incorporate DistilBERT, a smaller, faster, cheaper, and lighter version of BERT, for efficient sentiment analysis.
   - **GPT-3/GPT-4**: Leverage advanced models like GPT-3 or GPT-4 for zero-shot, few-shot, or fine-tuned sentiment analysis to capture more complex patterns in the data.

2. **Hybrid Embeddings**
   - Combine different embedding methods (e.g., BERT embeddings with TF-IDF features) to capture both contextual and statistical information, enhancing model performance.

2. **Contextualized Embeddings**:
   - Consider using newer models like T5 (Text-to-Text Transfer Transformer) or XLNet, which have shown improvements over BERT in some tasks.

### Advanced Classifiers and Techniques

1. **Ensemble Methods**
   - **Stacking**: Combine multiple classifiers using a meta-classifier to improve performance and leverage the strengths of different models.
   - **Blending**: Use blending techniques where predictions from multiple models are combined to make the final prediction, enhancing overall accuracy.

2. **Deep Learning Models**
   - **Recurrent Neural Networks (RNNs) with Attention Mechanism**: Implement RNNs with attention mechanisms to focus on important words or phrases in the tweets.
   - **Convolutional Neural Networks (CNNs)**: Use CNNs to capture local features in text data, which can improve the model's ability to identify key patterns.
   - **Transformers**: Utilize full transformer-based models directly for classification tasks to take advantage of their powerful context awareness.

### Data Augmentation

1. **Synthetic Data Generation**
   - Apply methods like SMOTE for text data to generate synthetic examples, especially for balancing the dataset and addressing class imbalance issues. Although our dataset is balanced, these techniques can be useful for other datasets with class imbalance.

2. **Back Translation**
   - Translate tweets to another language and then back to English to create more training data with slight variations, enhancing model robustness.

3. **Additional Data Cleaning and Augmentation Techniques**
   - Ensure comprehensive preprocessing steps, such as handling emojis, URLs, mentions, and special characters in tweets.
   - Besides back translation, consider using techniques like Easy Data Augmentation (EDA), which includes operations like synonym replacement, random insertion, etc.

### Handling Imbalanced Data

- Techniques like undersampling, oversampling, or using class weights in your loss function can help manage class imbalances effectively. Even though our dataset has no imbalance, if it had, we would need to handle it using these methods.

### Model Optimization

- Use techniques like Grid Search or Random Search for hyperparameter tuning to optimize your models' performance for both classification and text embedding extraction.

### Evaluation and Interpretation

1. **Model Interpretability**
   - **LIME (Local Interpretable Model-agnostic Explanations)**: Use LIME to understand the model's predictions on individual tweets, providing insights into model behavior.
   - **SHAP (SHapley Additive exPlanations)**: Utilize SHAP to provide both global and local interpretations of model outputs, ensuring transparency and trust in the model.

2. **Robustness Testing**
   - Test models against adversarial examples to check robustness and ensure reliability under various conditions.
   - Evaluate the model on different datasets to assess generalization and adaptability to new data.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setting Up a Virtual Environment

```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```


### Installing Dependencies

```
git clone https://github.com/ardaxz99/Twitter-Sentiment-Analysis-A-Complete-Guide-to-Text-Classification.git
cd Twitter-Sentiment-Analysis
pip install -r requirements.txt
```

### Downloading the Dataset

1. Download the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) as a CSV file named `training.1600000.processed.noemoticon.csv`.
2. Copy the downloaded CSV file to the working directory of the project.



## Usage

Execute the Jupyter notebook to reproduce the results:

```
jupyter nbconvert --to notebook --execute main.ipynb
```

## Contributors

- **Arda Baris Basaran**

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
