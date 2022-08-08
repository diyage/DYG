from abc import abstractmethod


class BaseEvaluator:
    def __init__(
            self,
    ):
        pass

    @abstractmethod
    def make_targets(
            self,
            *args,
            **kwargs
    ):
        pass

    @abstractmethod
    def eval_detector_mAP(
            self,
            *args,
            **kwargs
    ):
        pass

