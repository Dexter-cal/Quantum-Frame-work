"""
qai.nlp -- Natural Language Processing & Tokenization Utilities.
"""
import re
from typing import List, Dict, Any
import numpy as np

def text_cleaner(text: str) -> str:
    """Cleans raw text (lower-casing, removing punctuation & extra whitespace)."""
    clean = re.sub(r'[^\w\s]', '', text.lower())
    return re.sub(r'\s+', ' ', clean).strip()

def tf_idf_vectorizer(corpus: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Computes Term Frequency-Inverse Document Frequency (TF-IDF) matrix."""
    clean_corpus = [text_cleaner(doc).split() for doc in corpus]
    vocab = sorted(list(set(word for doc in clean_corpus for word in doc)))
    vocab_map = {word: idx for idx, word in enumerate(vocab)}

    N = len(corpus)
    tf = np.zeros((N, len(vocab)), dtype=float)
    df = np.zeros(len(vocab), dtype=float)

    for i, doc in enumerate(clean_corpus):
        for word in doc:
            col = vocab_map[word]
            tf[i, col] += 1.0
        if len(doc) > 0:
            tf[i, :] /= len(doc)
        unique_words = set(doc)
        for word in unique_words:
            df[vocab_map[word]] += 1.0

    idf = np.log((N + 1.0) / (df + 1.0)) + 1.0
    tfidf = tf * idf
    return tfidf, vocab

class WordPieceTokenizer:
    """Subword WordPiece Tokenizer."""
    def __init__(self, vocab: List[str]):
        self.vocab = set(vocab)

    def tokenize(self, text: str) -> List[str]:
        words = text_cleaner(text).split()
        tokens = []
        for word in words:
            if word in self.vocab:
                tokens.append(word)
            else:
                tokens.append("[UNK]")
        return tokens

def word_embeddings_lookup(tokens: List[str], embedding_dim: int = 16) -> np.ndarray:
    """Maps token strings to deterministic pseudo-word embeddings."""
    vecs = []
    for token in tokens:
        seed = sum(ord(c) for c in token)
        np.random.seed(seed)
        vecs.append(np.random.randn(embedding_dim))
    return np.array(vecs)
