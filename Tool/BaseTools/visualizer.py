from abc import abstractmethod


class BaseVisualizer:
    def __init__(self):
        pass

    @abstractmethod
    def make_targets(
            self,
            *args,
            **kwargs
    ):
        pass

    @abstractmethod
    def detect_one_image(
            self,
            *args,
            **kwargs
    ):
        pass


