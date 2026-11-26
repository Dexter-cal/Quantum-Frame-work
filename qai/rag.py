"""
qai.rag -- next-generation Retrieval-Augmented Generation (2024-2026 era).

Five RAG paradigms in one module:

  1. BaselineRAG    -- standard vector-similarity top-k retrieval
  2. GraphRAG       -- Microsoft 2024: build a knowledge graph + community
                       summaries; answer global questions via map-reduce
  3. SelfRAG        -- Asai 2023: train reflection tokens that decide when
                       to retrieve, grade relevance, and verify support
  4. CorrectiveRAG  -- CRAG: retrieval evaluator + knowledge-strip grading
                       + web-search fallback
  5. HyDE           -- hypothetical document embeddings: ask the LLM to
                       hallucinate an answer, embed it, retrieve neighbours
  6. RAGFusion      -- multi-query fusion: expand the user query into N
                       paraphrases, retrieve for each, RRF-fuse the results

All retrievers share a common interface so they can be swapped in
qai.build(technique="...", rag="graphrag").

This is a self-contained implementation that uses numpy for vector math
(no external vector DB needed) so it works out of the box.
"""
from __future__ import annotations
import hashlib
import math
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
import numpy as np


# ============================================================================
# Common utilities
# ============================================================================
def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _simple_embed(text: str, dim: int = 64, vocab: Optional[Dict[str, int]] = None) -> np.ndarray:
    """Bag-of-words hashing embedding (deterministic, no model needed)."""
    vec = np.zeros(dim, dtype=np.float32)
    toks = _tokenize(text)
    for t in toks:
        if vocab is not None and t in vocab:
            vec[vocab[t] % dim] += 1.0
        else:
            # hash trick
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
            vec[(h >> 8) % dim] -= 0.5
    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec /= norm
    return vec


def _build_vocab(docs: List[str]) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for d in docs:
        for t in _tokenize(d):
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def reciprocal_rank_fusion(ranked_lists: List[List[str]],
                            k: int = 60) -> List[Tuple[str, float]]:
    """RRF: combine multiple ranked lists into a single ranking."""
    scores: Dict[str, float] = defaultdict(float)
    for rlist in ranked_lists:
        for rank, doc in enumerate(rlist):
            scores[doc] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


# ============================================================================
# 1. BASELINE RAG (vector-similarity top-k)
# ============================================================================
class BaselineRAG:
    """Standard vector-similarity RAG with a configurable embedder."""

    def __init__(self, documents: List[str],
                 embedder: Optional[Callable[[str], np.ndarray]] = None,
                 dim: int = 64):
        self.documents = list(documents)
        self.embedder = embedder or (lambda t: _simple_embed(t, dim))
        self.embeddings = np.array([self.embedder(d) for d in self.documents])

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        q = self.embedder(query)
        scores = [_cosine(q, e) for e in self.embeddings]
        idx = np.argsort(scores)[-top_k:][::-1]
        return [(self.documents[i], float(scores[i])) for i in idx if scores[i] > 1e-6]

    def generate(self, query: str, top_k: int = 3,
                  llm: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
        retrieved = self.retrieve(query, top_k=top_k)
        context = "\n---\n".join(d for d, _ in retrieved)
        if llm is None:
            answer = f"[RAG] Based on {len(retrieved)} retrieved chunks: {context[:200]}..."
        else:
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            answer = llm(prompt)
        return {"query": query, "answer": answer, "retrieved": retrieved}


# ============================================================================
# 2. GRAPH RAG (Microsoft 2024)
# ============================================================================
class _Entity:
    def __init__(self, name: str):
        self.name = name
        self.description = ""
        self.documents: Set[int] = set()
        self.relationships: Dict[str, List[str]] = defaultdict(list)  # type -> entities


class GraphRAG:
    """Microsoft GraphRAG (Edge et al. 2024).

    Two-stage pipeline:
      Index:
        1. Extract (entity, relation) tuples from each document via simple NER
        2. Cluster entities into communities via label propagation
        3. Generate a community summary (here: top entities + relations)
      Query:
        - global: ask each community to produce a partial answer,
                  map-reduce into a final answer
        - local:  start from entities mentioned in the query, BFS their
                  neighbourhood, gather the relevant context
    """

    def __init__(self, documents: List[str],
                 embedder: Optional[Callable[[str], np.ndarray]] = None,
                 dim: int = 64, n_communities: int = 5):
        self.documents = list(documents)
        self.embedder = embedder or (lambda t: _simple_embed(t, dim))
        self.entities: Dict[str, _Entity] = {}
        self.doc_embeddings = np.array([self.embedder(d) for d in self.documents])
        self.community_of: Dict[str, int] = {}
        self.community_summaries: Dict[int, str] = {}
        self.n_communities = n_communities
        self._index()

    def _index(self) -> None:
        """Extract entities + relations from each document."""
        for i, doc in enumerate(self.documents):
            # naive NER: capitalised words + noun-phrase-ish patterns
            toks = re.findall(r"\b[A-Z][a-z]+\b", doc)
            # add noun-phrase-like sequences (word, word) pairs as relations
            for j, tok in enumerate(toks):
                if tok not in self.entities:
                    self.entities[tok] = _Entity(tok)
                self.entities[tok].documents.add(i)
                if j + 1 < len(toks):
                    nxt = toks[j + 1]
                    self.entities[tok].relationships["next"].append(nxt)
                    if nxt not in self.entities:
                        self.entities[nxt] = _Entity(nxt)
                    self.entities[nxt].documents.add(i)
        self._cluster_communities()

    def _cluster_communities(self) -> None:
        """Cluster entities via simple label propagation on the co-occurrence graph."""
        names = list(self.entities.keys())
        if not names:
            return
        idx = {n: i for i, n in enumerate(names)}
        n = len(names)
        # build adjacency via shared documents
        adj = np.zeros((n, n), dtype=np.float32)
        for a, ea in self.entities.items():
            for b, eb in self.entities.items():
                if a == b:
                    continue
                shared = len(ea.documents & eb.documents)
                if shared > 0:
                    adj[idx[a], idx[b]] = shared
        # label propagation
        labels = np.arange(n) % self.n_communities
        for _ in range(10):
            new_labels = labels.copy()
            for i in range(n):
                if adj[i].sum() == 0:
                    continue
                neighbours = np.where(adj[i] > 0)[0]
                votes = labels[neighbours]
                counts = np.bincount(votes, minlength=self.n_communities)
                new_labels[i] = counts.argmax()
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
        for name, label in zip(names, labels):
            self.community_of[name] = int(label)
        # build community summaries
        for c in range(self.n_communities):
            ents = [n for n, l in self.community_of.items() if l == c]
            if not ents:
                continue
            doc_ids = sorted(set().union(*[self.entities[e].documents for e in ents]))
            summary = f"Community {c}: entities={ents[:5]}... spans docs {doc_ids[:3]}"
            self.community_summaries[c] = summary

    def global_search(self, query: str,
                       llm: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
        """Map-reduce over all community summaries."""
        partials = []
        for cid, summary in self.community_summaries.items():
            if llm is None:
                partials.append({"community": cid, "answer": f"[partial {cid}] {summary}"})
            else:
                prompt = f"Community summary:\n{summary}\n\nQuestion: {query}\nPartial answer:"
                partials.append({"community": cid, "answer": llm(prompt)})
        if llm is None:
            final = f"[GraphRAG global] {len(partials)} communities contributed. Top: {partials[0]['answer'] if partials else 'N/A'}"
        else:
            joined = "\n".join(p["answer"] for p in partials)
            final = llm(f"Partial answers:\n{joined}\n\nSynthesise a final answer for: {query}")
        return {"query": query, "answer": final, "partials": partials}

    def local_search(self, query: str,
                      llm: Optional[Callable[[str], str]] = None,
                      hops: int = 2) -> Dict[str, Any]:
        """Start from query entities, BFS their neighbours."""
        seed_ents = [e for e in self.entities if e.lower() in query.lower()]
        visited: Set[str] = set()
        frontier = list(seed_ents)
        for _ in range(hops):
            new_frontier = []
            for e in frontier:
                if e in visited:
                    continue
                visited.add(e)
                for relation_targets in self.entities[e].relationships.values():
                    new_frontier.extend(t for t in relation_targets if t not in visited)
            frontier = new_frontier
        # gather documents from visited entities
        doc_ids = sorted(set().union(*[self.entities[e].documents for e in visited]))
        context = "\n---\n".join(self.documents[i][:200] for i in doc_ids[:5])
        if llm is None:
            answer = f"[GraphRAG local] entities visited={list(visited)[:5]} docs={doc_ids[:3]}"
        else:
            answer = llm(f"Context:\n{context}\n\nQuestion: {query}\nAnswer:")
        return {"query": query, "answer": answer, "entities_visited": list(visited),
                "doc_ids": doc_ids}

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """For drop-in compatibility with BaselineRAG."""
        result = self.local_search(query)
        return [(self.documents[i], 1.0) for i in result["doc_ids"][:top_k]]


# ============================================================================
# 3. SELF-RAG (Asai et al. 2023)
# ============================================================================
class SelfRAG:
    """Self-RAG: reflection tokens gate retrieval + generation.

    Tokens:
      [Retrieve]   -- yes/no: should we retrieve?
      [ISREL]      -- relevant/irrelevant for each retrieved doc
      [ISSUP]      -- fully / partially / not supported by the doc
      [ISUSE]      -- utility score 1-5 of the response

    A simple rule-based reflection layer is used by default; pass a real
    `critic` callable (LLM) for production use.
    """

    def __init__(self, documents: List[str],
                 embedder: Optional[Callable[[str], np.ndarray]] = None,
                 critic: Optional[Callable[[str, str], str]] = None,
                 dim: int = 64, retrieval_threshold: float = 0.3):
        self.documents = list(documents)
        self.embedder = embedder or (lambda t: _simple_embed(t, dim))
        self.embeddings = np.array([self.embedder(d) for d in self.documents])
        self.critic = critic
        self.retrieval_threshold = retrieval_threshold

    def _should_retrieve(self, query: str) -> bool:
        """Heuristic: retrieve if the query contains a factual/question word."""
        triggers = {"what", "who", "when", "where", "why", "how", "explain",
                    "describe", "list", "compare", "define"}
        toks = _tokenize(query)
        return any(t in triggers for t in toks)

    def _grade_relevance(self, query: str, doc: str) -> str:
        if self.critic:
            return self.critic(f"Is this doc relevant to the query?\nQ: {query}\nDoc: {doc[:200]}\nAnswer relevant or irrelevant:", "relevant")
        # heuristic: token overlap
        q_toks = set(_tokenize(query))
        d_toks = set(_tokenize(doc))
        overlap = len(q_toks & d_toks) / max(len(q_toks), 1)
        return "relevant" if overlap > self.retrieval_threshold else "irrelevant"

    def _grade_support(self, doc: str, generation: str) -> str:
        if self.critic:
            return self.critic(f"Is the generation fully supported by the doc?\nDoc: {doc[:200]}\nGen: {generation[:200]}\nAnswer fully, partially, or no:", "fully")
        gen_toks = set(_tokenize(generation))
        doc_toks = set(_tokenize(doc))
        if not gen_toks:
            return "no"
        coverage = len(gen_toks & doc_toks) / len(gen_toks)
        if coverage > 0.7:
            return "fully"
        if coverage > 0.3:
            return "partially"
        return "no"

    def _utility(self, query: str, generation: str) -> int:
        # 1-5 score based on length + presence of question words
        score = min(5, max(1, len(generation.split()) // 20 + 1))
        return score

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        if not self._should_retrieve(query):
            return []
        q = self.embedder(query)
        scores = [_cosine(q, e) for e in self.embeddings]
        idx = np.argsort(scores)[-top_k:][::-1]
        out = []
        for i in idx:
            if scores[i] <= 1e-6:
                continue
            rel = self._grade_relevance(query, self.documents[i])
            if rel == "relevant":
                out.append((self.documents[i], float(scores[i]), rel))
        return out

    def generate(self, query: str, top_k: int = 3,
                  llm: Optional[Callable[[str], str]] = None,
                  max_retries: int = 1) -> Dict[str, Any]:
        retrieved = self.retrieve(query, top_k=top_k)
        if not retrieved and max_retries > 0:
            # retry with a rephrased query (very simple)
            return self.generate(query + " ?", top_k=top_k, llm=llm, max_retries=0)
        context = "\n---\n".join(d for d, _, _ in retrieved)
        if llm is None:
            answer = f"[SelfRAG] answer using {len(retrieved)} relevant docs"
        else:
            answer = llm(f"Context:\n{context}\n\nQuestion: {query}\nAnswer:")
        # verify support
        support = self._grade_support(context, answer) if retrieved else "no"
        utility = self._utility(query, answer)
        return {
            "query": query,
            "answer": answer,
            "retrieved": retrieved,
            "support": support,
            "utility": utility,
            "should_retrieve": self._should_retrieve(query),
        }


# ============================================================================
# 4. CORRECTIVE RAG (CRAG)
# ============================================================================
class CorrectiveRAG:
    """Corrective RAG (Yan et al. 2024).

    Pipeline:
      1. Retrieve top-k from vector store
      2. Retrieval evaluator grades each doc: correct / ambiguous / incorrect
      3. If any 'correct' -> refine knowledge strips
      4. If all 'ambiguous' / 'incorrect' -> web search fallback
      5. Generate final answer with refined context
    """

    def __init__(self, documents: List[str],
                 embedder: Optional[Callable[[str], np.ndarray]] = None,
                 web_search: Optional[Callable[[str], List[str]]] = None,
                 dim: int = 64,
                 confidence_threshold: float = 0.5):
        self.documents = list(documents)
        self.embedder = embedder or (lambda t: _simple_embed(t, dim))
        self.embeddings = np.array([self.embedder(d) for d in self.documents])
        self.web_search = web_search
        self.confidence_threshold = confidence_threshold

    def _evaluate(self, query: str, doc: str) -> str:
        """Return 'correct', 'ambiguous', or 'incorrect'."""
        q = self.embedder(query)
        d = self.embedder(doc)
        sim = _cosine(q, d)
        if sim > self.confidence_threshold + 0.2:
            return "correct"
        if sim > self.confidence_threshold:
            return "ambiguous"
        return "incorrect"

    def _refine(self, doc: str, query: str) -> str:
        """Partition doc into sentences, keep only relevant ones."""
        sentences = re.split(r"(?<=[.!?])\s+", doc)
        q_toks = set(_tokenize(query))
        kept = []
        for s in sentences:
            s_toks = set(_tokenize(s))
            if len(q_toks & s_toks) / max(len(q_toks), 1) > 0.2:
                kept.append(s)
        return " ".join(kept) if kept else doc

    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        q = self.embedder(query)
        scores = [_cosine(q, e) for e in self.embeddings]
        idx = np.argsort(scores)[-top_k:][::-1]
        retrieved = []
        for i in idx:
            if scores[i] <= 1e-6:
                continue
            grade = self._evaluate(query, self.documents[i])
            retrieved.append({"doc": self.documents[i], "score": float(scores[i]),
                              "grade": grade})
        # decision: web search fallback?
        has_correct = any(r["grade"] == "correct" for r in retrieved)
        web_results: List[str] = []
        if not has_correct and self.web_search is not None:
            web_results = self.web_search(query)[:3]
        # refine knowledge strips
        refined = [self._refine(r["doc"], query) for r in retrieved
                   if r["grade"] in ("correct", "ambiguous")]
        return {
            "retrieved": retrieved,
            "web_fallback_used": not has_correct and self.web_search is not None,
            "web_results": web_results,
            "refined_context": "\n".join(refined + web_results),
        }

    def generate(self, query: str, top_k: int = 5,
                  llm: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
        retrieval = self.retrieve(query, top_k=top_k)
        context = retrieval["refined_context"]
        if llm is None:
            answer = f"[CRAG] used web fallback={retrieval['web_fallback_used']} context_len={len(context)}"
        else:
            answer = llm(f"Context:\n{context}\n\nQuestion: {query}\nAnswer:")
        return {"query": query, "answer": answer, **retrieval}


# ============================================================================
# 5. HyDE -- Hypothetical Document Embeddings (Gao et al. 2022)
# ============================================================================
class HyDE:
    """HyDE: hallucinate a hypothetical answer, embed it, retrieve neighbours.

    Useful when query and document distributions differ a lot (e.g. short
    keyword queries vs long detailed passages).
    """

    def __init__(self, documents: List[str],
                 embedder: Optional[Callable[[str], np.ndarray]] = None,
                 hypo_generator: Optional[Callable[[str], str]] = None,
                 dim: int = 64):
        self.documents = list(documents)
        self.embedder = embedder or (lambda t: _simple_embed(t, dim))
        self.embeddings = np.array([self.embedder(d) for d in self.documents])
        self.hypo = hypo_generator or self._default_hypo

    @staticmethod
    def _default_hypo(query: str) -> str:
        """Pretend to answer the query with a generic template."""
        return f"The answer to '{query}' involves several key considerations. First, relevant concepts must be identified. Second, supporting evidence is gathered. Finally, a synthesis is presented."

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        hypo = self.hypo(query)
        q = self.embedder(hypo)
        scores = [_cosine(q, e) for e in self.embeddings]
        idx = np.argsort(scores)[-top_k:][::-1]
        return [(self.documents[i], float(scores[i])) for i in idx if scores[i] > 1e-6]

    def generate(self, query: str, top_k: int = 3,
                  llm: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
        retrieved = self.retrieve(query, top_k=top_k)
        context = "\n---\n".join(d for d, _ in retrieved)
        if llm is None:
            answer = f"[HyDE] hypo='{self.hypo(query)[:80]}...' retrieved={len(retrieved)}"
        else:
            answer = llm(f"Context:\n{context}\n\nQuestion: {query}\nAnswer:")
        return {"query": query, "answer": answer, "retrieved": retrieved,
                "hypothesis": self.hypo(query)}


# ============================================================================
# 6. RAG-FUSION -- multi-query + RRF (Radev 2002 / Wang 2024)
# ============================================================================
class RAGFusion:
    """RAG-Fusion: expand query into N paraphrases, retrieve for each, RRF-fuse."""

    def __init__(self, documents: List[str],
                 embedder: Optional[Callable[[str], np.ndarray]] = None,
                 query_expander: Optional[Callable[[str], List[str]]] = None,
                 dim: int = 64, n_expansions: int = 4):
        self.documents = list(documents)
        self.embedder = embedder or (lambda t: _simple_embed(t, dim))
        self.embeddings = np.array([self.embedder(d) for d in self.documents])
        self.expander = query_expander or self._default_expander
        self.n_expansions = n_expansions

    @staticmethod
    def _default_expander(query: str) -> List[str]:
        """Simple heuristic expansions."""
        toks = _tokenize(query)
        expansions = [query]
        if len(toks) > 1:
            expansions.append(" ".join(reversed(toks)))
            expansions.append(" ".join(toks[:len(toks)//2]) + " " + " ".join(toks[len(toks)//2:]))
        expansions.append("explain " + query)
        expansions.append(query + " in detail")
        return expansions[:4]

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        expansions = self.expander(query)
        ranked_lists = []
        for q in expansions:
            qv = self.embedder(q)
            scores = [_cosine(qv, e) for e in self.embeddings]
            order = np.argsort(scores)[-top_k:][::-1]
            ranked_lists.append([self.documents[i] for i in order if scores[i] > 1e-6])
        fused = reciprocal_rank_fusion(ranked_lists)
        # attach documents
        doc_to_score = {d: s for d, s in fused[:top_k]}
        return [(d, s) for d, s in doc_to_score.items()]

    def generate(self, query: str, top_k: int = 3,
                  llm: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
        retrieved = self.retrieve(query, top_k=top_k)
        context = "\n---\n".join(d for d, _ in retrieved)
        if llm is None:
            answer = f"[RAG-Fusion] expansions={self.expander(query)} fused={len(retrieved)}"
        else:
            answer = llm(f"Context:\n{context}\n\nQuestion: {query}\nAnswer:")
        return {"query": query, "answer": answer, "retrieved": retrieved,
                "expansions": self.expander(query)}


# ============================================================================
# Unified registry
# ============================================================================
RAG_REGISTRY: Dict[str, type] = {
    "baseline": BaselineRAG,
    "graphrag": GraphRAG,
    "selfrag": SelfRAG,
    "crag": CorrectiveRAG,
    "hyde": HyDE,
    "fusion": RAGFusion,
}


def build_rag(kind: str, documents: List[str], **kwargs) -> Any:
    """Factory: build_rag('graphrag', docs) -> GraphRAG instance."""
    if kind not in RAG_REGISTRY:
        raise ValueError(f"Unknown RAG '{kind}'. Available: {sorted(RAG_REGISTRY.keys())}")
    return RAG_REGISTRY[kind](documents, **kwargs)


__all__ = [
    "BaselineRAG", "GraphRAG", "SelfRAG", "CorrectiveRAG", "HyDE", "RAGFusion",
    "RAG_REGISTRY", "build_rag", "reciprocal_rank_fusion",
]
