import numpy as np
from Tool.FCNDesNet101_SS.Predictor import SSPredictor
from Tool.FCNDesNet101_SS.Tools import SSTools
from Tool.FCNDesNet101_SS.Model import FCNResnet101
from torch.utils.data import DataLoader
from tqdm import tqdm


class SSEvaluator:
    def __init__(
            self,
            model: FCNResnet101,
            predictor: SSPredictor,

    ):
        self.detector = model
        self.predictor = predictor
        self.device = next(model.parameters()).device

    def make_targets(
            self,
            labels: list
    ):
        targets = SSTools.make_target(
            labels
        )
        return targets.to(self.device)

    def eval_semantic_segmentation_accuracy(
            self,
            data_loader_test: DataLoader,
            desc: str = 'eval semantic segmentation accuracy',
    ):
        acc_vec_include_background = []
        acc_vec = []
        for batch_id, (images, objects_vec, masks_vec) in enumerate(tqdm(data_loader_test,
                                                                         desc=desc,
                                                                         position=0)):

            self.detector.eval()
            images = images.to(self.device)

            targets = self.make_targets(masks_vec)

            output = self.detector(images)

            gt_decode = self.predictor.decode_target(targets)  # type: np.ndarray
            pre_decode = self.predictor.decode_predict(output)  # type: np.ndarray

            pre_mask_vec = np.argmax(pre_decode, axis=-1)
            gt_mask_vec = np.argmax(gt_decode, axis=-1)

            acc = np.mean((pre_mask_vec == gt_mask_vec).astype(np.float32))
            acc_vec_include_background.append(acc)
            """
                do not consider background, it will cause very high accuracy !!
            """
            except_background = gt_mask_vec != 0
            acc = np.mean((pre_mask_vec[except_background] == gt_mask_vec[except_background]).astype(np.float32))
            acc_vec.append(acc)

        print('\nsemantic segmentation accuracy:{:.2%}, {:.2%}(include background)'.format(
            np.mean(acc_vec),
            np.mean(acc_vec_include_background)
        ))
