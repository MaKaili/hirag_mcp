from abc import ABC, abstractmethod
from typing import Callable, List
from dataclasses import dataclass

@dataclass
class BaseEntity(ABC):
    extract_func: Callable
    entity_extract_prompt: str

    @abstractmethod
    def entity(self, text: str) -> List[str]:
        pass
