from hirag_mcp.storage.base_gdb import BaseGDB
import networkx as nx
from hirag_mcp.schema import Relation, Entity
from typing import List
import asyncio

class NetworkXGDB(BaseGDB):
    path: str
    graph: nx.DiGraph
    llm_func: Callable

    def create(cls, path: str, llm_func: Callable):
        if not os.path.exists(path):
            graph = nx.Graph()
        else:
            graph = nx.read_gpickle(path)
        return cls(path=path, graph=graph, llm_func=llm_func)
    
    async def async_query_node(self, node_id: str) -> Entity:
        return self.graph.nodes[node_id]
    
    async def _async_upsert_node(self, node: Entity, record_description: Optional[str] = None) -> Optional[str]:
        """
        Upsert a node into the graph.
        
        This method adds a new node to the graph if it doesn't exist, or updates an existing node.
        For concurrent upsertion, we use the following strategy:
        If the node not in the graph, add it. Use the database's transaction atomic to 
        ensure the consistency of the graph.
        If the node in the graph, we record the description which we use to update the current node.
        If the record_description is the same as the description in the graph, we update the node, otherwise 
        return the description in the graph, to generate the new description.
        
        Args:
            node (Entity): The entity node to be inserted or updated
            record_description (Optional[str]): Description to compare with existing node's description
            
        Returns:
            Optional[str]: If the node exists and has a different description, returns the existing description.
                            Otherwise returns None.

        """
        if node.id not in self.graph.nodes:
            try:
                self.graph.add_node(node.id, {**node.metadata, "entity_name": node.page_content})
                return
            except Exception as e:
                # TODO: handle the exception
                raise e
        else:
            node = await self.query_node(node.id)
            latest_description = node.metadata.description
            if record_description == latest_description:
                self.graph.nodes[node.id].update({**node.metadata, "entity_name": node.page_content})
                return
            else:
                return latest_description

    async def _merge_node(self, node: Entity, latest_description: str) -> Entity:
        description_list = [node.metadata.description, latest_description]
        
        
    async def async_upsert_node(self, node: Entity):
        while True:
            latest_description = await self._async_upsert_node(node)
            if latest_description is None:
                break
            else:
                node = await self._merge_node(node, latest_description)
                
    async def async_upsert_nodes(self, nodes: List[Entity]):
       await asyncio.gather(*[self.upsert_node(node) for node in nodes])
        
    async def async_upsert_relation(self, relation: Relation):
        try:
            self.graph.add_edge(relation.source, relation.target, **relation.properties)
        except Exception as e:
            # TODO: handle the exception
            raise e

    async def async_upsert_relations(self, relations: List[Relation]):
        await asyncio.gather(*[self.upsert_relation(relation) for relation in relations])

    async def async_query_one_hop(self, node_id: str) -> (List[Entity], List[Relation]):
        return list(self.graph.neighbors(node_id)), list(self.graph.edges(node_id))