"""
Implementation of paper
["Mimicking Word Embeddings using Subword RNNs"](https://arxiv.org/abs/1707.06961)
using Polyglot word embeddings.
MIMIC allows to avoid OOV (out of vocabulary) problem by imitating the original
pre-trained word embeddings using small character-based model, this making
`<UNK>` word embeddings unnecessary. Benefits
(in comparison with FastText, for example): very low resource
requirements and significant boost in accuracy due to the absence of `<UNK>`
in your training data.
"""

import os.path
import random
from typing import List
import argparse

import numpy as np
from polyglot.mapping import Embedding as PolyEmbedding
from keras import backend as K, Input, optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras.models import Model, load_model

TEST_SET_SIZE = 1000
NUM_CHARS = 1 << 16


class MimicEmbedding:

    @classmethod
    def load(cls, polyglot_embedding_path: str,
             mimic_model_path: str):
        e = PolyEmbedding.load(polyglot_embedding_path)
        model = load_model(mimic_model_path, compile=False)
        return cls(e, model)

    def __init__(self, embedding: PolyEmbedding, mimic_model: Model):
        self.embedding = embedding
        self.mimic_model = mimic_model
        self.max_word_length = K.int_shape(mimic_model.input)[1]
        self.embedding_size = K.int_shape(mimic_model.output)[1]
        assert len(embedding.zero_vector()) == self.embedding_size

    def truncate_word(self, w: str) -> str:
        return (w if len(w) <= self.max_word_length
                else w[:self.max_word_length])

    def get_vectors(self, words: List[str]) -> np.ndarray:
        result = []
        unknown_words = []
        unknown_positions = []
        for position, w in enumerate(words):
            if w in self.embedding:
                word_index = self.embedding.vocabulary[w]
                result.append(self.embedding.vectors[word_index])
            else:
                unknown_words.append(self.truncate_word(w))
                unknown_positions.append(position)
                result.append(None)
        if unknown_words:
            inputs = words_to_inputs(unknown_words, self.max_word_length)
            predictions = self.mimic_model.predict_on_batch(inputs)
            for vector, position in zip(predictions, unknown_positions):
                result[position] = vector
        return np.asarray(result)

    def mimic_only_vectors(self, words: List[str]) -> np.ndarray:
        inputs = words_to_inputs(
            list(map(self.truncate_word, words)),
            self.max_word_length)
        predictions = self.mimic_model.predict_on_batch(inputs)
        return np.asarray(predictions)

    def nearest_to_vector(self, vector: np.ndarray, top_k: int) -> List[str]:
        sim_word_indices = (
            np.linalg.norm(self.embedding.vectors - vector, axis=-1)
            .argsort()[:top_k])
        return [self.embedding.words[i] for i in sim_word_indices]


def pre_trained_mimic_model_path(language: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'pretrained', 'mimic', language + '.h5')


def mse_loss(y_true, y_pred):
    return K.mean(K.sum(K.square(y_pred - y_true), axis=-1), axis=-1)


def create_mimic_model(max_word_length: int, embedding_size: int) -> Model:
    emb = Embedding(NUM_CHARS, 20, input_length=max_word_length,
                    embeddings_initializer='glorot_uniform',
                    name='mimic_emb')
    word_characters = Input(shape=(max_word_length,), dtype='int32',
                            name='mimic_word')
    char_embeddings = emb(word_characters)
    rnn_out = (
        Bidirectional(LSTM(64),
                      merge_mode='concat', name='mimic_rnn')
        (char_embeddings))
    dense1 = (
        Dense(100, activation='selu', name='mimic_dense1')
        (rnn_out))
    dense2 = (
        Dense(embedding_size, activation=None, name='mimic_dense2')
        (dense1))
    model = Model(inputs=word_characters, outputs=dense2)
    return model


def words_to_inputs(words: List[str], input_length: int) -> np.ndarray:
    result = np.zeros((len(words), input_length), dtype='int32')
    for i, word in enumerate(words):
        char_indices = list(map(ord, word))
        result[i, :len(char_indices)] = char_indices
    return result


def compose_dataset(full_embedding: PolyEmbedding, input_length: int):
    words = []
    vectors = []
    data = list(zip(full_embedding.words, full_embedding.vectors))
    random.shuffle(data)
    for word, vector in data:
        if len(word) > input_length:
            word = word[:input_length]
        words.append(word)
        vectors.append(vector)
    inputs = words_to_inputs(words, input_length)
    return inputs, np.array(vectors)


def train_mimic_model(polyglot_embedding_path: str,
                      mimic_model_path: str,
                      max_word_length: int,
                      num_epochs: int,
                      learning_rate: float,
                      use_dev_set: bool):
    full_embedding = PolyEmbedding.load(str(polyglot_embedding_path))
    embedding_size = len(full_embedding.zero_vector())
    all_X, all_Y = compose_dataset(full_embedding, max_word_length)
    if use_dev_set:
        train_X = all_X[TEST_SET_SIZE:]
        train_Y = all_Y[TEST_SET_SIZE:]
        validation_data = (all_X[:TEST_SET_SIZE], all_Y[:TEST_SET_SIZE])
    else:
        train_X, train_Y = all_X, all_Y
        validation_data = None
    model = create_mimic_model(max_word_length, embedding_size)
    optimizer = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer, loss=mse_loss)
    if os.path.exists(mimic_model_path):
        model.load_weights(mimic_model_path)
    loss_to_monitor = 'val_loss' if use_dev_set else 'loss'
    save_model = ModelCheckpoint(
        mimic_model_path, verbose=1, monitor=loss_to_monitor,
        save_best_only=True)
    lr_reducer = ReduceLROnPlateau(
        verbose=1, factor=0.2, min_lr=1e-7, monitor=loss_to_monitor,
        cooldown=100)
    model.fit(train_X, train_Y,
              batch_size=1024, epochs=num_epochs,
              callbacks=[save_model, lr_reducer],
              validation_data=validation_data)


def contain_tf_gpu_mem_usage():
    """
    By default TensorFlow may try to reserve all available GPU memory
    making it impossible to train multiple agents at once.
    This function will disable such behaviour in TensorFlow.
    """
    try:
        # noinspection PyPackageRequirements
        import tensorflow as tf
    except ImportError:
        pass
    else:
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory
        sess = tf.Session(config=config)
        set_session(sess)


def main():
    _parser = argparse.ArgumentParser(description='Trains MIMIC model')
    _parser.add_argument(
        '--epochs', type=int, default=10000, help='Number of epochs to train')
    _parser.add_argument(
        '--maxlen', type=int, default=20, help='Max word length')
    _parser.add_argument(
        '--save', type=str, required=True, metavar='PATH',
        help='Path where the model should be saved')
    _parser.add_argument(
        '--lr', type=float, default=1e-3, help='Initial learning rate')
    _parser.add_argument(
        '--devset', action='store_true',
        help=('Enables training with a separate dev '
              'set to evaluate performance'))
    _parser.add_argument(
        '--embeddings', type=str, required=True,
        metavar='PATH', help='Path to polyglot embeddings')
    _options = _parser.parse_args()
    random.seed(0)
    contain_tf_gpu_mem_usage()
    train_mimic_model(
        _options.embeddings, _options.save, _options.maxlen,
        _options.epochs, _options.lr, _options.devset)


if __name__ == '__main__':
    main()
