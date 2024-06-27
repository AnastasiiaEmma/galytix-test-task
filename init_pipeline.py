import gensim
from gensim.models import KeyedVectors

def load_and_save_word2vec():
    """
    Loads the 1st Mil word vectors from the binary file and saves them in a CSV format.
    """
    # Loading the first million word vectors from the binary file
    wv = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True, limit=1000000)
    
    # Saving the vectors in a flat file (CSV) format
    wv.save_word2vec_format('vectors.csv')

if __name__ == "__main__":
    load_and_save_word2vec()