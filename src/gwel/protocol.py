from abc import ABC, abstractmethod
import numpy as np


class Exporter(ABC):

    @abstractmethod
    def export(self, path: str):
        pass


