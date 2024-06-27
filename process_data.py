import pandas as pd
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import logging
import logging.config

# Configure logging
logging.config.fileConfig('logging.conf')

# Get a logger
logger = logging.getLogger('my_python_app')

def get_phrase_embedding(phrase, wv):
    """
    Gets the embedding for a given phrase by averaging the embeddings of the words in the phrase.

    Parameters:
    phrase (str): The phrase to get the embedding for.
    wv (KeyedVectors): The Word2Vec model.

    Returns:
    np.ndarray or None: The averaged embedding for the phrase or None if no words in the phrase are found in the Word2Vec model.
    """
    words = phrase.split()
    word_vectors = [wv[word] for word in words if word in wv]
    if word_vectors:
        logger.info(f"Embedding found for phrase: {phrase}")
        return sum(word_vectors) / len(word_vectors)
    else:
        logger.warning(f"No embedding found for phrase: {phrase}")
        return None
    
# Loading word vectors
logger.info("Loading word vectors")
wv = KeyedVectors.load_word2vec_format('vectors.csv', binary=False)

# Loading phrases
logger.info("Loading phrases")
phrases = pd.read_csv('phrases.csv', encoding='latin1')

# Assigning embeddings to phrases
phrases['embedding'] = phrases['Phrases'].apply(lambda x: get_phrase_embedding(x, wv))

# Dropping phrases with no embeddings
phrases = phrases.dropna(subset=['embedding'])

# Calculating distances
embeddings = np.array(phrases['embedding'].tolist())
distances = cosine_distances(embeddings)

# Saving the distances
distances_df = pd.DataFrame(distances, index=phrases['Phrases'], columns=phrases['Phrases'])
distances_df.to_csv('phrase_distances.csv')
logger.info("Saved phrase distances to phrase_distances.csv")

def find_closest_match(user_input, phrases, wv):
    """
    Finds the closest matching phrase in the data based on cosine distance.

    Parameters:
    user_input (str): The phrase input by the user.
    phrases (pd.DataFrame): DataFrame containing the phrases and their embeddings.
    wv (KeyedVectors): The Word2Vec model.

    Returns:
    tuple: The closest phrase and the distance to the closest phrase.
    """
    user_embedding = get_phrase_embedding(user_input, wv)
    if user_embedding is None:
        logger.error(f"No embedding found for user input: {user_input}")
        return None, None
    distances = phrases['embedding'].apply(lambda x: cosine_distances([user_embedding], [x])[0][0])
    closest_idx = distances.idxmin()
    closest_phrase = phrases.loc[closest_idx, 'Phrases']
    logger.info(f"Closest phrase to '{user_input}' is '{closest_phrase}' with distance {distances.min()}")
    return closest_phrase, distances.min()

# Example usage of find_closest_match
user_input = "banana split"
closest_phrase, distance = find_closest_match(user_input, phrases, wv)
print(f"Closest phrase: {closest_phrase} with distance {distance}")