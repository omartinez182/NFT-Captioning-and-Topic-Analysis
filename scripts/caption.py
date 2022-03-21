import torch
import numpy as np
import pickle
import os
from torchvision import transforms
from PIL import Image
import sys
sys.path.insert(0, 'scripts')
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    """
    Performs address (string) normalization.
    Params
    ------
    image_path: str
        Path to the image to be processed.

    Output
    ------
    image: array
        Transformed image array.
    """
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image

def get_caption(image, embed_size=256, hidden_size=512, num_layers=1):
    """
    Performs address (string) normalization.
    Params
    ------
    image:
        Image to be processed.

    embed_size:
        Size of the embeddings.
    
    hidden_size:
        Hidden size of NN.

    num_layers:
        Number of layers.

    Output
    ------
    sentence: str
        Description of the image.

    image_path: str
        Name of the image.
    """
    image_path = image

    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open('models/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(embed_size).eval()
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load('models/encoder-5-3000.pkl'))
    decoder.load_state_dict(torch.load('models/decoder-5-3000.pkl'))

    # Prepare an image
    image = load_image(image, transform)
    image_tensor = image.to(device)

    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    return sentence, image_path