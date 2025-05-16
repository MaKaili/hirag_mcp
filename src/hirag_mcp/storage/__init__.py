from .lancedb import LanceDB
from .base_vdb import BaseVDB
from typing import Literal

__all__ = ["LanceDB", "BaseVDB"]

DEFAULT_STORAGE_CONFIGS = {
    "lancedb": {
        "loader": LanceDB,
        "init_args": {
            "db_url": "test.db"
        }
    }
}

async def get_storage(
    storage_name: Literal["lancedb"],
    storage_configs: dict,
) -> BaseVDB:
    if storage_name == "lancedb":
        return await LanceDB.create(
            embedding_func=storage_configs["embedding_func"],
            db_url=storage_configs["db_url"],
            strategy_provider=storage_configs["strategy_provider"]
        )
    else:
        raise ValueError(f"Invalid storage name: {storage_name}")