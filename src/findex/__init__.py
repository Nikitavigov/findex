"""findex: Пошуковий рушій, розроблений з нуля на Python."""

from findex.boolean import merge_and, merge_not, merge_or, set_and, set_not, set_or
from findex.corpus import Document, iter_documents
from findex.index import (
    Index,
    InvertedIndex,
    build_index,
    load_index,
    open_index,
    save_index,
)
from findex.models import DocMeta, Posting, SearchResult
from findex.query import And, Not, Or, Phrase, QueryNode, Term, parse_query
from findex.ranking import BM25Scorer, Scorer, TfIdfScorer, rank_documents
from findex.search import execute_query, search_ranked, timed
from findex.snippets import generate_snippet
from findex.tokenizer import normalize_text, tokenize

__all__ = [
    "And",
    "BM25Scorer",
    "DocMeta",
    "Document",
    "Index",
    "InvertedIndex",
    "Not",
    "Or",
    "Phrase",
    "Posting",
    "QueryNode",
    "Scorer",
    "SearchResult",
    "Term",
    "TfIdfScorer",
    "build_index",
    "execute_query",
    "generate_snippet",
    "iter_documents",
    "load_index",
    "merge_and",
    "merge_not",
    "merge_or",
    "normalize_text",
    "open_index",
    "parse_query",
    "rank_documents",
    "save_index",
    "search_ranked",
    "set_and",
    "set_not",
    "set_or",
    "timed",
    "tokenize",
]
