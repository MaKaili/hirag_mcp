import asyncio
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from hirag_mcp._llm import gpt_4o_mini_complete, openai_embedding
from hirag_mcp._utils import _limited_gather
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

    _chunk_pool: ProcessPoolExecutor | None = None

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

    @classmethod
    def _get_pool(cls) -> ProcessPoolExecutor:
        if cls._chunk_pool is None:
            ctx = multiprocessing.get_context("spawn")
            cpu = os.cpu_count() or 1
            cls._chunk_pool = ProcessPoolExecutor(
                max_workers=cpu,
                mp_context=ctx,
            )
        return cls._chunk_pool

    chunk_upsert_concurrency: int = 4
    entity_upsert_concurrency: int = 4
    relation_upsert_concurrency: int = 2

    async def _process_document(self, document):
        loop = asyncio.get_running_loop()
        pool = self._get_pool()
        chunks = await loop.run_in_executor(pool, self.chunker.chunk, document)

        chunk_coros = [
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
        await _limited_gather(chunk_coros, self.chunk_upsert_concurrency)

        entities = await self.entity_extractor.entity(chunks)
        entity_coros = [
            self.vdb.upsert_text(
                text_to_embed=ent.metadata.description,
                properties={
                    "document_key": ent.id,
                    "text": ent.page_content,
                    **ent.metadata.__dict__,
                },
                table_name="entities",
                mode="overwrite",
            )
            for ent in entities
        ]
        await _limited_gather(entity_coros, self.entity_upsert_concurrency)

        relations = await self.entity_extractor.relation(chunks, entities)
        relation_coros = [self.gdb.upsert_relation(rel) for rel in relations]
        await _limited_gather(relation_coros, self.relation_upsert_concurrency)

    async def insert_to_kb(
        self,
        document_path: str,
        content_type: str,
        document_meta: Optional[dict] = None,
        loader_configs: Optional[dict] = None,
    ):
        start_total = time.perf_counter()
        # change to a async function
        documents = await asyncio.to_thread(
            load_document,
            document_path,
            content_type,
            document_meta,
            loader_configs,
        )

        tasks = [self._process_document(doc) for doc in documents]

        await asyncio.gather(*tasks)
        total = time.perf_counter() - start_total
        logger.info(f"Total pipeline time: {total:.3f}s")
