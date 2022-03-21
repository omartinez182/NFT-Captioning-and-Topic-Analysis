import os
import sys
import pandas as pd
sys.path.insert(0, 'scripts')
import caption
from caption import get_caption
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN


data = {}
count = 0
dir = 'data/raw/'
for image in os.listdir(dir):
    count += 1
    print("Processing image #"+str(count))
    img = get_caption(dir+image)
    data[img[1].replace('data/raw/','')] = img[0]

dct = {k:[v] for k,v in data.items()} 
df = pd.DataFrame.from_dict(dct, orient='index') 
df.reset_index(inplace=True)
df.columns = ['image_name', 'caption']
df.caption=df.caption.str.replace("<start>","", regex=True)
df.caption=df.caption.str.replace("<end>","", regex=True)
df.caption=df.caption.str.replace(".","", regex=True)
df.to_csv('data/outputs/captions.csv')