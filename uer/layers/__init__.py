from uer.layers.embeddings import WordEmbedding
from uer.layers.embeddings import WordPosEmbedding
from uer.layers.embeddings import WordPosSegEmbedding
from uer.layers.embeddings import WordSinusoidalposEmbedding
from uer.layers.embeddings import WordPosSegTimeEmbedding


str2embedding = {"word": WordEmbedding, "word_pos": WordPosEmbedding, "word_pos_seg": WordPosSegEmbedding,
                 "word_sinusoidalpos": WordSinusoidalposEmbedding, "word_pos_seg_time": WordPosSegTimeEmbedding}

__all__ = ["WordEmbedding", "WordPosEmbedding", "WordPosSegEmbedding",
           "WordSinusoidalposEmbedding", "str2embedding", "WordPosSegTimeEmbedding"]
