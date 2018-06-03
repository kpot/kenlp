import os.path

from kelp.mimic import pre_trained_mimic_model_path, MimicEmbedding


def test_mimic_embeddings():
    mem = MimicEmbedding.load(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'fixtures', 'en_embeddings_pkl.tar.bz2'),
        pre_trained_mimic_model_path('en'))
    assert len(mem.truncate_word('long_unknown' * 20)) == mem.max_word_length
    word_vecs = mem.get_vectors(['cluster', 'long_unknown' * 20])
    assert word_vecs.shape == (2, mem.embedding_size)
    sim_words = mem.nearest_to_vector(word_vecs[0], 10)
    assert len(sim_words) == 10
