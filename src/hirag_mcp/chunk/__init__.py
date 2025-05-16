from typing import Literal, Optional

from langchain_core.documents import Document

from .base_chunk import BaseChunk
from .fix_token_chunk import FixTokenChunk

DEFAULT_CHUNK_MAP = {
    "fix_token": {
        "chunker": FixTokenChunk,
        "init_args": {"chunk_size": 1000, "chunk_overlap": 200},
    }
}


def chunk_document(
    document: Document,
    chunk_configs: Optional[dict] = None,
    chunk_type: Literal["fix_token"] = "fix_token",
) -> list[Document]:
    if chunk_configs is None:
        chunk_configs = DEFAULT_CHUNK_MAP[chunk_type]

    chunker = chunk_configs["chunker"](**chunk_configs["init_args"])
    return chunker.chunk(document)


def chunk_documents(
    documents: list[Document],
    chunk_configs: Optional[dict] = None,
    chunk_type: Literal["fix_token"] = "fix_token",
) -> list[Document]:
    return [
        chunk for document in documents for chunk in chunk_document(document, chunk_configs, chunk_type)
    ]


__all__ = ["FixTokenChunk", "chunk_document", "chunk_documents", "BaseChunk"]
