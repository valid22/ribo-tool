import functools
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

MAX_SEQUENCE_LENGTH = 256
CHARACTER_MAPPING = {'R' : 'G', 'Y' : 'T', 'M' : 'A', 'K' : 'G', 'S' : 'G', 'W' : 'A', 'H' : 'A', 'B' : 'G', 'V' : 'G', 'D' : 'G'}
TOKEN_MAPPING = "ATGCN"

classifier_model = load_model('./models/classifier.h5')
detector_model = load_model('./models/detector.h5')

def character_mapping(x):
    x = functools.reduce(lambda a, kv: a.replace(*kv), CHARACTER_MAPPING.items(), x.upper())
    return x

tk = Tokenizer(num_words=5, char_level=True)
tk.fit_on_texts("ATGCN")

def process_sequence(seq, for_classifier=False):
    seq = character_mapping(seq)

    if for_classifier:
        seq = np.array([[int(letter_to_index(e)) for e in seq]])
    else:
        seq = np.array([seq])
        seq = tk.texts_to_sequences(seq)

    return pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH if not for_classifier else 250)


def letter_to_index(letter):
    if letter not in TOKEN_MAPPING:
        print ("Letter not present")
        print (letter)
    return next((i for i, _letter in enumerate(TOKEN_MAPPING) if _letter == letter), None)
