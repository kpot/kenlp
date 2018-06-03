NLP tools for Keras
===================
A set of tools for natural language processing using Keras.

MIMIC embeddings
----------------
Implementation of paper ["Mimicking Word Embeddings using Subword RNNs"](https://arxiv.org/abs/1707.06961).
MIMIC allows to avoid OOV (out of vocabulary) problem by imitating the original
pre-trained word embeddings using small character-based model, this making
`<UNK>` word embeddings unnecessary. Benefits
(in comparison with FastText, for example): very low resource
requirements and significant boost in accuracy due to the absence of `<UNK>`
in your training data.

A short example which can mimic word embeddings for OOV words and show
which of the known words are closest to the produced vector:

    from kelp.embeddings import MimicEmbedding
    from pathlib import Path
    
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

The example above being run with some fictional words:

    Enter a word: dunamite
    'dunamite' is an unknown word which has to be mimicked
    similar words: ['phosphine', 'superoxide', 'hydrazine', 'phospholipid', 'histidine', 'trypsin', 'oxalate', 'surfactant', 'biotin', 'acetaldehyde']

    Enter a word: Kaa
    'Kaa' is an unknown word which has to be mimicked
    similar words: ['Ung', 'Tham', 'Kuang', 'Chon', 'Teh', 'Kor', 'Kum', 'Gam', 'Loh', 'Mah']
    
    Enter a word: Karll
    'Karll' is an unknown word which has to be mimicked
    similar words: ['Helmer', 'Gerhardt', 'Bastian', 'Henrich', 'Tilman', 'Bosse', 'Laurin', 'Hartwig', 'Burkhard', 'Cori']
    
    Enter a word: abroktose
    'abroktose' is an unknown word which has to be mimicked
    similar words: ['olefin', 'analyte', 'epsilon', 'reactant', 'granule', 'aerosol', 'alkene', 'anionic', 'chloroplast', 'erythrocyte']

The word "dunamite" is a misspelled "dynamite", but MIMIC "understands" that this
is probably some high-energy and explosive material.
It was also able to guess that Kaa looks like an asian name, and "Karll" is something european.
"abroktose" turns out to be something vaguely scientific, probably about chemistry or biology,
which also makes sense.

To train a MIMIC model for a new language you will need to first [download
polyglot's embedding for that language](http://polyglot.readthedocs.io/en/latest/Download.html)
and then run a command like this:

    python -m kelp.mimic --save new_mimic_model.h5 --embeddings <path to downloaded polyglot embedding>

