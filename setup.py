import pandas as pd
import numpy as np
from bertopic import BERTopic

# Data preparation
captions = pd.read_csv('./data/processed/captions.csv')
captions.drop(['Unnamed: 0'], axis=1, inplace=True)
captions.caption = captions.caption.str.replace("<start>", "", regex=True)
captions.caption = captions.caption.str.replace("<end>", "", regex=True)
captions.caption = captions.caption.str.replace(".", "", regex=True)
docs = list(captions.caption.values)

# Model training
topic_model = BERTopic(min_topic_size=15, language="english",
                       calculate_probabilities=True, verbose=True)
topics, probs = topic_model.fit_transform(docs)

# Visualization
fig_intertopic = topic_model.visualize_topics()
fig_topics = topic_model.visualize_barchart(top_n_topics=10)

# Save output
fig_intertopic.write_html("intertopic_distance.html")
fig_topics.write_html("top_topics.html")

