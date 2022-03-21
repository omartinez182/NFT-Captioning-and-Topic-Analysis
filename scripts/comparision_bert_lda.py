# !pip install umap-learn
# !pip install stop_words
# !pip install sentence-transformers

from stop_words import get_stop_words
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap.umap_ as umap
import string
import time
from gensim import corpora
import gensim
from nltk.tokenize import word_tokenize
import nltk
import re
from sentence_transformers import SentenceTransformer


# In[2]:

ntopic = 20
nltk.download('punkt')
stop_words = get_stop_words('en')


# In[2]:


df = pd.read_csv('../data/captions.csv')
df.caption = df.caption.str.replace("<start>", "")
df.caption = df.caption.str.replace("<end>", "")
df.caption = df.caption.str.replace(".", "")
df.head()


# In[ ]:


# Append sentences to raw sentences
raw_sentences = []
for s in list(df.caption.values):
    raw_sentences.append(s)


# In[ ]:
def only_letters(tested_string):
    for letter in tested_string:
        if letter not in "abcdefghijklmnopqrstuvwxyz":
            return False
    return True


def clean_data(s):
    s = s.replace(">", "").lower()
    if "lines:" in s:
        index = s.index("lines:")
        s = s[index+10:]
    word_list = word_tokenize(s)
    cleaned = []
    for w in word_list:
        if w not in stop_words:
            if w in string.punctuation or only_letters(w):
                if w in string.punctuation or len(set(w)) > 1:
                    cleaned.append(w)
    return " ".join(cleaned), cleaned

# from documents clean sentence and return vocabulary of sentence


def build_data(docs):
    n_docs = len(docs)
    sentences = []  # sentences
    token_lists = []  # words vocabulary
    for i in range(len(docs)):
        sentence, token_list = clean_data(docs[i])
        if token_list:
            sentences.append(sentence)
            token_lists.append(token_list)
    return sentences, token_lists


# In[ ]:

sentences, token_lists = build_data(raw_sentences)

# In[ ]:


def predict_topics_with_kmeans(embeddings, num_topics):
    kmeans_model = KMeans(num_topics)
    kmeans_model.fit(embeddings)
    topics_labels = kmeans_model.predict(embeddings)
    return topics_labels

# In[ ]:


def plot_embeddings(embedding, labels, title):

    labels = np.array(labels)
    distinct_labels = set(labels)
    n = len(embedding)
    counter = Counter(labels)
    for i in range(len(distinct_labels)):
        ratio = (counter[i] / n) * 100
        cluster_label = f"cluster {i}: { round(ratio,2)}"
        x = embedding[:, 0][labels == i]
        y = embedding[:, 1][labels == i]
        plt.plot(x, y, '.', alpha=0.4, label=cluster_label)
    plt.legend(title="Topic", loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.title(title)


# In[ ]:


def reduce_umap(embedding):
    reducer = umap.UMAP()
    embedding_umap = reducer.fit_transform(embedding)
    return embedding_umap


# In[ ]:

def get_document_topic_lda(model, corpus, k):
        n_doc = len(corpus)
        # init a vector of size number of docs x clusters
        document_topic_mapping = np.zeros((n_doc, k))
        for i in range(n_doc):
            for topic, prob in model.get_document_topics(corpus[i]):
                document_topic_mapping[i, topic] = prob
        return document_topic_mapping


# In[ ]:


dictionary = corpora.Dictionary(token_lists)
corpus = [dictionary.doc2bow(text) for text in token_lists]
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=ntopic,
                                           id2word=dictionary, passes=20)

embedding_lda = get_document_topic_lda(ldamodel, corpus, ntopic)


# This is not a document embedding , I just use it as it is.
print("LDA vector shape :", embedding_lda.shape)


# In[ ]:


for i, topic in enumerate(embedding_lda[10].flatten()):
    print("Topic ", i+1, ") ", embedding_lda[10].flatten()[i])

# In[ ]:

ldamodel.get_document_topics(corpus[0])
srtd = sorted(ldamodel.get_document_topics(corpus[0]),
              key=lambda x: x[1], reverse=True)
print(srtd)
print(srtd[0][0])


# In[ ]:

labels_lda = []
for line in corpus:
    line_labels = sorted(ldamodel.get_document_topics(line),
                         key=lambda x: x[1], reverse=True)
    # 1st 0 is for selecting top item, and 2nd 0 is for index of tuple
    top_topic = line_labels[0][0]
    labels_lda.append(top_topic)


# In[ ]:

# Plot for LDA
embedding_umap_lda = reduce_umap(embedding_lda)
plot_embeddings(embedding_umap_lda, labels_lda, "LDA with Umap")

# In[ ]:
# Metrics for LDA
print("Silhouette score:")
print("LDA with UMAP : ", silhouette_score(embedding_umap_lda, labels_lda))

# In[ ]:

# Plot for BERT
model_bert = SentenceTransformer('bert-base-nli-max-tokens')
embedding_bert = np.array(model_bert.encode(sentences, show_progress_bar=True))
embedding_umap_bert = reduce_umap(embedding_bert)
labels_bert_umap = predict_topics_with_kmeans(embedding_umap_bert, ntopic)
plot_embeddings(embedding_umap_bert, labels_bert_umap, "Bert with Umap")


# In[ ]:

# Metrics for BERT
print("Silhouette score:")
print("Bert with Umap", silhouette_score(embedding_umap_bert,
                                         labels_bert_umap))
