from abc import ABC, abstractmethod
from hirag_mcp.schema import Relation, Entity
from typing import List

class BaseGDB(ABC):
    @abstractmethod
    async def upsert_nodes(self, nodes: List[Entity]):
        raise NotImplementedError

    @abstractmethod
    async def upsert_relations(self, relations: List[Relation]):
        raise NotImplementedError

    @abstractmethod
    async def query_one_hop(self, query: str) -> (List[Entity], List[Relation]):
        raise NotImplementedError
