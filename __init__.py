# my_python_app/__init__.py

# Initialization code
print("Initializing my_python_app package")

# Imports functions and classes to make them accessible directly from the package
from .init_pipeline import load_and_save_word2vec
from .process_data import get_phrase_embedding, calculate_distances, find_closest_match
