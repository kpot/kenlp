"""
A short example which can mimic word embeddings for OOV words and show
which of the known words are closest to the produced vector.
"""

from pathlib import Path

from kenlp.mimic import MimicEmbedding, pre_trained_mimic_model_path

if __name__ == '__main__':
    mem = MimicEmbedding.load(
        str(Path.home() / 'polyglot_data/embeddings2/en/embeddings_pkl.tar.bz2'),
        pre_trained_mimic_model_path('en'))
    while True:
        w = input('Enter a word: ')
        if w not in mem.embedding:
            print(f'{w!r} is an unknown word which has to be mimicked')
        word_vec = mem.get_vectors([w])[-1]
        sim_words = mem.nearest_to_vector(word_vec, 10)
        print('similar words:', sim_words, '\n')
