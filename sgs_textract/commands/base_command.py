from abc import ABC, abstractmethod


class BaseCommand(ABC):
    @abstractmethod
    def add_arguments(self, parser):
        pass

    @abstractmethod
    def execute(self, args):
        pass
