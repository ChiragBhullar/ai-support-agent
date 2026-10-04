"""Retrieval over the FAQ and troubleshooting guides (the "R" in RAG).

Documents are Markdown files where every `## heading` starts a new entry. We use
character n-gram TF-IDF, which needs no API, runs in milliseconds for a support
knowledge base, and tolerates typos ("pasword" still matches "password").
Swapping in an embedding model + vector store only means replacing this class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    source: str


@dataclass(frozen=True)
class SearchHit:
    document: Document
    score: float


def load_markdown_sections(path: Path) -> list[Document]:
    docs: list[Document] = []
    title, body = None, []
    for line in path.read_text(encoding="utf-8").splitlines() + ["## "]:
        if line.startswith("## "):
            if title:
                docs.append(
                    Document(
                        id=f"{path.stem}-{len(docs) + 1}",
                        title=title,
                        text="\n".join(body).strip(),
                        source=path.name,
                    )
                )
            title, body = line[3:].strip(), []
        elif title is not None:
            body.append(line)
    return docs


class KnowledgeBase:
    def __init__(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("KnowledgeBase needs at least one document")
        self.documents = documents
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True
        )
        # Titles are the questions customers ask, so weight them twice.
        corpus = [f"{d.title} {d.title} {d.text}" for d in documents]
        self._matrix = self._vectorizer.fit_transform(corpus)

    @classmethod
    def from_markdown(cls, path: Path) -> KnowledgeBase:
        return cls(load_markdown_sections(path))

    def search(self, query: str, k: int = 3, threshold: float = 0.0) -> list[SearchHit]:
        scores = cosine_similarity(self._vectorizer.transform([query]), self._matrix)[0]
        ranked = scores.argsort()[::-1][:k]
        return [
            SearchHit(self.documents[i], float(scores[i])) for i in ranked if scores[i] >= threshold
        ]
