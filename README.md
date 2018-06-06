NLP tools for Keras
===================
A set of tools for natural language processing using Keras.

MIMICK embeddings
----------------
Implementation of paper ["Mimicking Word Embeddings using Subword RNNs"](https://arxiv.org/abs/1707.06961).
MIMICK allows to avoid OOV (out of vocabulary) problem by imitating the original
pre-trained word embeddings using small character-based model, thus making
`<UNK>` word embeddings unnecessary. Benefits
(in comparison with FastText, for example): very low resource
requirements and significant boost in accuracy due to the absence of `<UNK>`
in your training data.

A short example which can mimic word embeddings for OOV words and show
which of the known words are closest to the produced vector:

    from kenlp.mimic import MimicEmbedding, pre_trained_mimic_model_path
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

    Enter a word: trintiful
    'trintiful' is an unknown word which has to be mimicked
    similar words: ['banal', 'cliched', 'preposterous', 'horrifying', 'horrid', 'baffling', 'disagreeable', 'pedantic', 'farcical', 'ludicrous']
    
    Enter a word: unfrogable
    'unfrogable' is an unknown word which has to be mimicked
    similar words: ['unmanageable', 'unwelcome', 'unbreakable', 'irreplaceable', 'unsatisfying', 'anachronistic', 'unrestrained', 'outmoded', 'unfashionable', 'unattainable']

    Enter a word: Kaa
    'Kaa' is an unknown word which has to be mimicked
    similar words: ['Ung', 'Tham', 'Kuang', 'Chon', 'Teh', 'Kor', 'Kum', 'Gam', 'Loh', 'Mah']
    
    Enter a word: Karll
    'Karll' is an unknown word which has to be mimicked
    similar words: ['Helmer', 'Gerhardt', 'Bastian', 'Henrich', 'Tilman', 'Bosse', 'Laurin', 'Hartwig', 'Burkhard', 'Cori']
    
    Enter a word: abroktose
    'abroktose' is an unknown word which has to be mimicked
    similar words: ['phosphine', 'adenine', 'albumin', 'oxalate', 'osmium', 'IgG', 'capsaicin', 'casein', 'amide', 'azide']

The word "trintiful" is fictional but MIMICK "understands" that this
is probably some kind of adjective. It assumed that "unfrogable"
is a negating adjective. It was also able to guess that Kaa looks like
an asian name, and "Karll" is something european. "abroktose" turns out
to be something vaguely scientific, probably about chemistry or biology,
which also makes sense.


The code already contains [pre-trained models for few languages](kenlp/pretrained/mimic)
(see the example on how to use them).
To train a MIMICK model for a new language you will need first to [download
polyglot's embedding for that language](http://polyglot.readthedocs.io/en/latest/Download.html)
and then run a command like this:

    python -m kenlp.mimic --save new_mimic_model.h5 --embeddings <path to downloaded polyglot embedding>


Installation
------------
Assuming you use virtualenv or venv, installation on Ubuntu >= 16.04 looks like

    sudo apt-get install libicu-dev
    git clone https://github.com/kpot/kenlp.git
    cd kenlp
    pip install .

On MacOS:

    brew install icu4c
    brew link --force icu4c
    git clone https://github.com/kpot/kenlp.git
    cd kenlp
    pip install .
    brew unlink icu4c

You will also need to install any of Keras's backends, like TensorFlow:

    pip install tensorflow

