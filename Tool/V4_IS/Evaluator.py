import numpy as np
from Tool.V4_IS.Predictor import YOLOV4PredictorIS
from Tool.V4_IS.Tools import YOLOV4ToolsIS
from Tool.V4_IS.Model import YOLOV4ForISModel
from Tool.V4.Evaluator import YOLOV4Evaluator
from torch.utils.data import DataLoader
from tqdm import tqdm


class YOLOV4EvaluatorIS(YOLOV4Evaluator):
    def __init__(
            self,
            model: YOLOV4ForISModel,
            predictor: YOLOV4PredictorIS,
            iou_th_for_make_target: float,
            multi_gt: bool,
    ):
        super().__init__(
            model,
            predictor,
            iou_th_for_make_target,
            multi_gt,
        )

    def change_image_wh(
            self,
            image_wh: tuple
    ):
        self.image_size = image_wh
        self.grid_number, self.pre_anchor_w_h = YOLOV4ToolsIS.get_grid_number_and_pre_anchor_w_h(
            self.image_size,
            self.image_shrink_rate,
            self.pre_anchor_w_h_rate
        )

    def make_targets(
            self,
            labels
    ):
        targets = YOLOV4ToolsIS.make_target(
            labels,
            self.pre_anchor_w_h,
            self.image_size,
            self.grid_number,
            self.kinds_name,
            self.iou_th_for_make_target,
            multi_gt=self.multi_gt
        )
        targets['mask'] = targets['mask'].to(self.device)
        for anchor_key in self.anchor_keys:
            targets[anchor_key] = targets[anchor_key].to(self.device)
        return targets

    def eval_semantic_segmentation_accuracy(
            self,
            data_loader_test: DataLoader,
            desc: str = 'eval semantic segmentation accuracy',
    ):
        acc_vec = []
        for batch_id, (images, objects_vec, masks_vec) in enumerate(tqdm(data_loader_test,
                                                                         desc=desc,
                                                                         position=0)):

            self.detector.eval()
            images = images.to(self.device)

            labels = [objects_vec, masks_vec]
            targets = self.make_targets(labels)

            output = self.detector(images)

            gt_decode = self.predictor.decode_target(targets)  # [kps_vec, masks_vec]_s
            pre_decode = self.predictor.decode_predict(output)  # [kps_vec, masks_vec]_s

            for image_index in range(images.shape[0]):
                pre_mask_vec = pre_decode[image_index][1]  # type: np.ndarray
                gt_mask_vec = gt_decode[image_index][1]  # type: np.ndarray
                acc = np.mean((pre_mask_vec == gt_mask_vec).astype(np.float32))
                acc_vec.append(acc)

        print('\nsemantic segmentation accuracy:{:.2%}'.format(np.mean(acc_vec)))


def debug_evaluator():
    from Tool.V4_IS.Model import YOLOV4ForISModel, CSPDarkNet53IS
    from Tool.V4 import YOLOV4Config

    config = YOLOV4Config()

    backbone = CSPDarkNet53IS()
    net = YOLOV4ForISModel(backbone, 3, 20)

    predictor = YOLOV4PredictorIS(
        0.5,
        0.0,
        0.0,
        0.1,
        config.data_config.pre_anchor_w_h_rate,
        config.data_config.kinds_name,
        config.data_config.image_size,
        config.data_config.image_shrink_rate,
    )
    ev = YOLOV4EvaluatorIS(net, predictor, 0.5, True)
    from Tool.V4_IS.DatasetDefine import get_voc_for_all_tasks_loader

    loader = get_voc_for_all_tasks_loader(
        '/home/dell/data/DataSet/VOC/',
        ['2007', '2012'],
        True,
        608,
        trans_form=None,
        use_bbox=True,
        use_mask_type=-1
    )
    ev.eval_semantic_segmentation_accuracy(loader)


if __name__ == '__main__':
    pass
