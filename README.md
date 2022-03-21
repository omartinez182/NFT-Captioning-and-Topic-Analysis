# Identify NFT Trends
![image](https://github.com/Kimiria/AIPI540/blob/main/NFT.jpg)

## Problem Statement
 * The Non-Fungible Token (NFT) is a fresh topic full of heat that is gaining public attention.
 * The largest NFT marketplace OpenSea doesn’t have the function to reflect categorization by the elements on the NFT, so we decided to apply CV & NLP to analyze the current NFT trends. (i.e. What are current NFTs about?)
 * On the one hand, it helps to provide investment reference for buyers, and on the other hand, it also provides creation reference for sellers.

## Data Source & Data Process
[Kaggle](https://www.kaggle.com/datasets/vepnar/nft-art-dataset) dataset with 3000+ images from the 2021 NFT Collection

We then used a pre-trained model `data\nft_image_captioning.py` to generate captions for the images, producing a [new dataset](https://github.com/bkenan/nft_nlp/blob/main/data/captions.csv) of descriptions. 

## Getting Started

First install the requirements necessary to run the python files.

```
pip install -r requirements.txt
```

### Machine Learning Approach
Latent Dirichlet allocation (LDA) is a particularly popular method for topic modelling. It treats each sentence as a mix of topics and each topic as a mix of words. This method allowed us to find the overlapping content.

After installing all of the requirements, you can execute the file `?.py`

   * The evaluation metric will be the silhouette value, which is a measure of how similar an object is to its own cluster compared to other clusters


### Deep Learning Approach
BERTopic is a topic modeling technique that leverages BERT based transformers and TF-IDF to create dense clusters.
 * It is used for easily interpretable topics.
 * Keep important words in the topic descriptions.
 * Works well both on CPU and GPU.
 
So we use BERT to extract different embeddings based on the context of the words. And we used UMAP for dimensionality reduction due to its best performance among other methods. The we use TF-IDF to demonstrate the important words in the topics. You can execute the file `experiments\bert_lda.py`

  * The evaluation metric will be the silhouette value(0.5), which is much better than the ML's(-0.13).

