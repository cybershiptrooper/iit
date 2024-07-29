from abc import ABC, abstractmethod


class HLModel(ABC):

    @abstractmethod
    def is_categorical(self) -> bool:
        pass