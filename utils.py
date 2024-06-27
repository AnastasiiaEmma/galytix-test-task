import logging

# Get a logger
logger = logging.getLogger(__name__)

def safe_get_phrase_embedding(phrase, wv):
    """
    Safely gets the embedding for a given phrase by averaging the embeddings of the words in the phrase.

    Parameters:
    phrase (str): The phrase to get the embedding for.
    wv (KeyedVectors): The Word2Vec model.

    Returns:
    np.ndarray or None: The averaged embedding for the phrase or None if an error occurs.
    """
    try:
        words = phrase.split()
        word_vectors = [wv[word] for word in words if word in wv]
        if word_vectors:
            logger.info(f"Embedding found for phrase: {phrase}")
            return sum(word_vectors) / len(word_vectors)
        else:
            logger.warning(f"No embedding found for phrase: {phrase}")
            return None
    except Exception as e:
        logger.error(f"Error processing phrase '{phrase}': {e}")
        return None
