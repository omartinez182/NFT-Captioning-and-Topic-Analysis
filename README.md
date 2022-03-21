# Identify NFT Trends
![image](https://github.com/Kimiria/AIPI540/blob/main/NFT.jpg)

## Problem Statement
 * The Non-Fungible Tokens (NFTs) are a fresh topic full of heat that is gaining public attention.
 * The largest NFT marketplaces don't have the functionality to reflect categorization by the elements on the NFT, so we decided to apply CV & NLP to analyze the current NFT trends. (i.e. What are current NFTs about?)
 * This information, on the one hand, helps to provide investment references for buyers, and on the other hand, it also provides a creative reference for sellers.

## Getting Started
Our approach consists of first using a publicly available dataset of NFT images, and then performing image captioning on each of them, we then use the captions to do topic modeling and create clusters based on the topics. Notice that these clusters are based on the underlying topics identified in the image captioning process, and are vulnerable to errors in that process. Nonetheless, this process serves as a proof of concept on the idea of combining both CV and NLP to understand and analyze visual elements in natural language. 

## Data Source & Data Process
[Kaggle](https://www.kaggle.com/datasets/vepnar/nft-art-dataset) dataset with 3000+ images from the 2021 NFT Collection

We then used a pre-trained model `data\nft_image_captioning.py` to generate captions for the images, producing a [new dataset](https://github.com/bkenan/nft_nlp/blob/main/data/captions.csv) of descriptions. 

## Getting Started

First, install the requirements necessary to run the python files.

```
pip install -r requirements.txt
```
Then, because the trained model is large it is necessary to use Git Large File Storage (LFS) to pull the pre-trained model. If you don’t have LFS installed, you can do this with either brew or apt-get.

```
$ brew update 
$ brew install git-lfs
```
Or
```
$ apt-get install git-lfs
```
After that, you can simply do a lfs pull to get the large file.
```
$ git lfs pull
```
Now you can execute the file to create the captioning 

```
$ python3 scripts/make_dataset.py
```
Notice that here we provide a sample image as an example because the full dataset would take hours to be produced. However, you can include any additional images (or download the full dataset from Kaggle) to produce the captions for all images. For convenience, we’ve also included the output already to use for modeling, so this shouldn’t be necessary.

Finally, to do the modeling and create the topic clusters you can run:
```
$ python3 setup.py
```
The output will be 2 HTML files with the interactive plots for the topic clusters, and the bar plots for the most popular topics. You can directly use your browser to open these files.

## Modeling Details

For modeling, we tested two different approaches, one that uses a classic statistical/machine learning approach (LDA) and another one that leverages deep learning (BERT). The details of each approach are as follows:

### Machine Learning Approach
Latent Dirichlet allocation (LDA) is a particularly popular method for topic modeling. It treats each sentence as a mix of topics and each topic as a mix of words. This method allowed us to find the overlapping content.


### Deep Learning Approach
BERTopic is a topic modeling technique that leverages BERT-based transformers and TF-IDF to create dense clusters. Some of the benefits of this approach are:
 * It is used for easily interpretable topics.
 * Keep important words in the topic descriptions.
 * Works well both on CPU and GPU.
 
So we use BERT to extract different embeddings based on the context of the words. And we used UMAP for dimensionality reduction due to its best performance among other methods. Then we use TF-IDF to demonstrate the important words in each cluster (each topic becomes a document). 


