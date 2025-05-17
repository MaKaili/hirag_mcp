import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from hirag_mcp._llm import gpt_4o_mini_complete, openai_embedding
from hirag_mcp.chunk import BaseChunk, FixTokenChunk
from hirag_mcp.entity import BaseEntity, VanillaEntity
from hirag_mcp.loader import load_document
from hirag_mcp.storage import (
    BaseGDB,
    BaseVDB,
    LanceDB,
    NetworkXGDB,
    RetrievalStrategyProvider,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

logging.getLogger("HiRAG").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("HiRAG")
logger = logging.getLogger("HiRAG-BENCH")


@dataclass
class HiRAG:
    # Chunk documents
    chunker: BaseChunk = field(
        default_factory=lambda: FixTokenChunk(chunk_size=1200, chunk_overlap=200)
    )

    # Entity extraction
    entity_extractor: BaseEntity = field(
        default_factory=lambda: VanillaEntity.create(extract_func=gpt_4o_mini_complete)
    )

    # Storage
    vdb: BaseVDB = field(default=None)
    gdb: BaseGDB = field(
        default_factory=lambda: NetworkXGDB.create(
            path="kb/hirag.gpickle",
            llm_func=gpt_4o_mini_complete,
        )
    )

    @classmethod
    async def create(cls, **kwargs):
        if kwargs.get("vdb") is None:
            lancedb = await LanceDB.create(
                embedding_func=openai_embedding,
                db_url="kb/hirag.db",
                strategy_provider=RetrievalStrategyProvider(),
            )
            kwargs["vdb"] = lancedb
        return cls(**kwargs)

    async def _process_document(self, document):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            chunks = await loop.run_in_executor(pool, self.chunker.chunk, document)

        await asyncio.gather(
            *[
                self.vdb.upsert_text(
                    text_to_embed=chunk.page_content,
                    properties={
                        "document_key": chunk.id,
                        "text": chunk.page_content,
                        **chunk.metadata.__dict__,
                    },
                    table_name="chunks",
                    mode="overwrite",
                )
                for chunk in chunks
            ]
        )

        entities = await self.entity_extractor.entity(chunks)

        await asyncio.gather(
            *[
                self.vdb.upsert_text(
                    text_to_embed=entity.metadata.description,
                    properties={
                        "document_key": entity.id,
                        "text": entity.page_content,
                        **entity.metadata.__dict__,
                    },
                    table_name="entities",
                    mode="overwrite",
                )
                for entity in entities
            ]
        )

        relations = await self.entity_extractor.relation(chunks, entities)

        await asyncio.gather(*[self.gdb.upsert_relation(r) for r in relations])

    async def insert_to_kb(
        self,
        document_path: str,
        content_type: str,
        document_meta: Optional[dict] = None,
        loader_configs: Optional[dict] = None,
    ):
        start_total = time.perf_counter()
        documents = load_document(
            document_path, content_type, document_meta, loader_configs
        )

        tasks = [self._process_document(doc) for doc in documents]

        await asyncio.gather(*tasks)
        total = time.perf_counter() - start_total
        logger.info(f"Total pipeline time: {total:.3f}s")
