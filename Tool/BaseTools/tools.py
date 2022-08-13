'''
This packet is the most-most-most core development tool.
It will serve for all other development tools.
You could use it to define everything !!!
'''
import torch
import numpy as np
from Tool.BaseTools.cv2_ import CV2
from typing import Union
from abc import abstractmethod


class BaseTools:
    @staticmethod
    @abstractmethod
    def get_grid_number_and_pre_anchor_w_h(
            image_wh: tuple,
            image_shrink_rate: Union[dict, tuple],
            pre_anchor_w_h_rate: Union[dict, tuple]
    ):
        pass

    @staticmethod
    @abstractmethod
    def make_target(
            *args,
            **kwargs
    ):
        '''
        create target used for computing loss.
        You may have one question: where is make_predict ?
        Method --make_predict-- is just __call__(or forward, little incorrect) of nn.Module !!!
        So predict is just the output of model(nn.Module you define).
        Please see model.BaseModel
        '''
        pass

    @staticmethod
    @abstractmethod
    def split_target(
            *args,
            **kwargs
    ) -> dict:
        '''
        split target(create by method make_target).
        you will get result like this:
            {
                 'position': xxx,
                 'conf': xxx,
                 'cls_prob': xxx,
                 ...
            }
        or:
            {
                'key_0': {
                             'position': xxx,
                             'conf': xxx,
                             'cls_prob': xxx,
                             ...
                        },
                'key_1': {
                             'position': xxx,
                             'conf': xxx,
                             'cls_prob': xxx,
                             ...
                        },
                ...
            }
        '''
        pass

    @staticmethod
    @abstractmethod
    def split_predict(
            *args,
            **kwargs
    ) -> dict:
        '''
                split predict(create by model, i.e. model output).
                you will get result like this:
                    {
                         'position': xxx,
                         'conf': xxx,
                         'cls_prob': xxx,
                         ...
                    }
                or:
                    {
                        'key_0': {
                                     'position': xxx,
                                     'conf': xxx,
                                     'cls_prob': xxx,
                                     ...
                                },
                        'key_1': {
                                     'position': xxx,
                                     'conf': xxx,
                                     'cls_prob': xxx,
                                     ...
                                },
                        ...
                    }
                '''
        pass

    @staticmethod
    def image_np_to_tensor(
            image: np.ndarray,
            mean=[0.406, 0.456, 0.485],
            std=[0.225, 0.224, 0.229]
    ) -> torch.Tensor:
        # image is a BGR uint8 image
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)

        image = CV2.cvtColorToRGB(image)  # (H, W, C)
        image = ((1.0 * image / 255.0) - mean) / std  # (H, W, C)
        image = np.transpose(image, axes=(2, 0, 1))  # (C, H, W)
        return torch.tensor(image, dtype=torch.float32)

    @staticmethod
    def image_tensor_to_np(
            img: torch.Tensor,
            mean=[0.406, 0.456, 0.485],
            std=[0.225, 0.224, 0.229]
    ) -> np.ndarray:
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        img = img.cpu().detach().numpy().copy()  # type:np.ndarray
        # (C, H, W)
        img = np.transpose(img, axes=(1, 2, 0))  # type:np.ndarray
        # (H, W, C)
        img = ((img * std) + mean) * 255.0
        img = np.array(img, np.uint8)  # type:np.ndarray
        img = CV2.cvtColorToBGR(img)
        return img

    @staticmethod
    def compute_iou_m_to_n(
            bbox1: Union[torch.Tensor, np.ndarray, list],
            bbox2: Union[torch.Tensor, np.ndarray, list]
    ) -> torch.Tensor:

        if isinstance(bbox1, np.ndarray) or isinstance(bbox1, list):
            bbox1 = torch.tensor(bbox1)
            if len(bbox1.shape) == 1:
                bbox1 = bbox1.unsqueeze(0)

        if isinstance(bbox2, np.ndarray) or isinstance(bbox2, list):
            bbox2 = torch.tensor(bbox2)
            if len(bbox2.shape) == 1:
                bbox2 = bbox2.unsqueeze(0)

        N = bbox1.size(0)
        M = bbox2.size(0)

        # Compute left-top coordinate of the intersections
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Conpute right-bottom coordinate of the intersections
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute area of the intersections from the coordinates
        wh = rb - lt  # width and height of the intersection, [N, M, 2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

        # Compute area of the bboxes
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])  # [N, ]
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])  # [M, ]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M, ] -> [1, M] -> [N, M]

        # Compute IoU from the areas
        union = area1 + area2 - inter  # [N, M]
        iou = inter / union  # [N, M]
        return iou

    @staticmethod
    def iou_score(bboxes_a, bboxes_b):
        """
            bbox_1 : [B*N, 4] = [x1, y1, x2, y2]
            bbox_2 : [B*N, 4] = [x1, y1, x2, y2]
        """
        tl = torch.max(bboxes_a[..., :2], bboxes_b[..., :2])
        br = torch.min(bboxes_a[..., 2:], bboxes_b[..., 2:])
        area_a = torch.prod(bboxes_a[..., 2:] - bboxes_a[..., :2], dim=-1)
        area_b = torch.prod(bboxes_b[..., 2:] - bboxes_b[..., :2], dim=-1)

        en = (tl < br).type(tl.type()).prod(dim=-1)
        area_i = torch.prod(br - tl, dim=-1) * en  # * ((tl < br).all())

        area_union = area_a + area_b - area_i
        iou = area_i / (area_union + 1e-20)

        return iou

    @staticmethod
    def compute_iou(
            boxes0: Union[torch.Tensor, np.ndarray, list],
            boxes1: Union[torch.Tensor, np.ndarray, list]
    ) -> torch.Tensor:
        if isinstance(boxes0, np.ndarray) or isinstance(boxes0, list):
            boxes0 = torch.tensor(boxes0)
            boxes1 = torch.tensor(boxes1)
            if len(boxes0.shape) == 1:
                boxes0 = boxes0.unsqueeze(0)
                boxes1 = boxes1.unsqueeze(0)
        # -1 * 4
        # -1 is boxes number

        # 4 if (x, y, x, y)
        return BaseTools.iou_score(boxes0, boxes1)
        # w0 = boxes0[..., 2] - boxes0[..., 0]  # -1
        # h0 = boxes0[..., 3] - boxes0[..., 1]  # -1
        # s0 = w0 * h0  # -1
        #
        # w1 = boxes1[..., 2] - boxes1[..., 0]  # -1
        # h1 = boxes1[..., 3] - boxes1[..., 1]  # -1
        # s1 = w1 * h1  # -1
        #
        # boxes = torch.stack((boxes0, boxes1), dim=-1)  # # -1 * 4 * 2
        #
        # inter_boxes_a_b = torch.max(boxes[..., 0:2, :], dim=-1)[0]  # # -1 * 2
        #
        # inter_boxes_m_n = torch.min(boxes[..., 2:4, :], dim=-1)[0]  # # -1 * 2
        #
        # inter_boxes_w_h = inter_boxes_m_n - inter_boxes_a_b  # # -1 * 2
        # inter_boxes_w_h[inter_boxes_w_h < 0] = 0.0  # # -1 * 2
        # inter_boxes_s = inter_boxes_w_h[..., 0] * inter_boxes_w_h[..., 1]  # # -1
        #
        # union_boxes_s = s0 + s1 - inter_boxes_s
        # iou = inter_boxes_s / (union_boxes_s + 1e-8)
        # return iou.clamp_(0.0, 1.0)  # -1

    @staticmethod
    def get_grid(
            grid_number: tuple
    ):
        index = torch.tensor(list(range(grid_number[0])), dtype=torch.float32)
        grid_r, grid_c = torch.meshgrid(index, index)
        grid = torch.cat((grid_c.unsqueeze(-1), grid_r.unsqueeze(-1)), dim=-1)
        # H * W * 2
        return grid

    @staticmethod
    def nms(
            position_abs: torch.Tensor,
            scores: torch.Tensor,
            threshold: float = 0.5
    ):
        position_abs = position_abs.cpu().detach().numpy().copy()
        scores = scores.cpu().detach().numpy().copy()
        """"Pure Python NMS baseline."""
        x1 = position_abs[:, 0]  # xmin
        y1 = position_abs[:, 1]  # ymin
        x2 = position_abs[:, 2]  # xmax
        y2 = position_abs[:, 3]  # ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # 计算交集的左上角点和右下角点的坐标
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # 计算交集的宽高
            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            # 计算交集的面积
            inter = w * h

            # 计算交并比
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-20)
            # 滤除超过nms阈值的检测框
            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]

        return keep

    # def nms(
    #         position_abs: torch.Tensor,
    #         conf: torch.Tensor,
    #         threshold: float = 0.5
    # ):
    #     x1 = position_abs[:, 0]
    #     y1 = position_abs[:, 1]
    #     x2 = position_abs[:, 2]
    #     y2 = position_abs[:, 3]
    #     areas = (x2 - x1) * (y2 - y1)  # [N,]
    #     _, order = conf.sort(0, descending=True)
    #
    #     keep = []
    #     while order.numel() > 0:
    #         if order.numel() == 1:  # just one
    #             i = order.item()
    #             keep.append(i)
    #             break
    #         else:
    #             i = order[0].item()  # max conf
    #             keep.append(i)
    #
    #         # 计算box[i]与其余各框的IOU(思路很好)
    #         xx1 = x1[order[1:]].clamp(min=x1[i])  # [N-1,]
    #         yy1 = y1[order[1:]].clamp(min=y1[i])
    #         xx2 = x2[order[1:]].clamp(max=x2[i])
    #         yy2 = y2[order[1:]].clamp(max=y2[i])
    #         inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)  # [N-1,]
    #
    #         iou = inter / (areas[i] + areas[order[1:]] - inter)  # [N-1,]
    #         idx = (iou <= threshold).nonzero().squeeze()  # idx[N-1,] order[N,]
    #         if idx.numel() == 0:
    #             break
    #         order = order[idx + 1]  #
    #
    #     return torch.tensor(keep, dtype=torch.long)

    @staticmethod
    def calculate_pr(gt_num, tp_list, confidence_score):
        """
        calculate all p-r pairs among different score_thresh for one class, using `tp_list` and `confidence_score`.

        Args:
            gt_num (Integer): 某张图片中某类别的gt数量
            tp_list (List): 记录某张图片中某类别的预测框是否为tp的情况
            confidence_score (List): 记录某张图片中某类别的预测框的score值 (与tp_list相对应)

        Returns:
            recall
            precision

        """
        if gt_num == 0:
            return [0], [0]
        if isinstance(tp_list, (tuple, list)):
            tp_list = np.array(tp_list)
        if isinstance(confidence_score, (tuple, list)):
            confidence_score = np.array(confidence_score)

        assert len(tp_list) == len(confidence_score), "len(tp_list) and len(confidence_score) should be same"

        if len(tp_list) == 0:
            return [0], [0]

        sort_mask = np.argsort(-confidence_score)
        tp_list = tp_list[sort_mask]
        recall = np.cumsum(tp_list) / gt_num
        precision = np.cumsum(tp_list) / (np.arange(len(tp_list)) + 1)

        return recall.tolist(), precision.tolist()

    @staticmethod
    def voc_ap(rec, prec, use_07_metric=False):
        """Compute VOC AP given precision and recall. If use_07_metric is true, uses
        the VOC 07 11-point method (default:False).
        """
        if isinstance(rec, (tuple, list)):
            rec = np.array(rec)
        if isinstance(prec, (tuple, list)):
            prec = np.array(prec)
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    @staticmethod
    def get_pre_kind_name_tp_score_and_gt_num(
            pre_kind_name_pos_score: list,
            gt_kind_name_pos_score: list,
            kinds_name: list,
            iou_th: float = 0.5
    ):
        '''
        just used for one image(all predicted box information(s))
        Args:
            pre_kind_name_pos_score: [kps0, kps1, ...]   kps --> (kind_name, (x, y, x, y), score)
            gt_kind_name_pos_score:
            kinds_name: [kind_name0, kind_name1, ...]
            iou_th:

        Returns:
            (
                kind_tp_and_score,
                gt_num
            )
            kind_tp_and_score = [
                            [kind_name, is_tp(0.0/1.0), score],
                            ...
                            ]
            gt_num --> dict key is each kind_name, val is real gt_num(TP+FN) of this kind_name
        '''
        pre_kind_name_pos_score = sorted(
            pre_kind_name_pos_score,
            key=lambda s: s[2],
            reverse=True
        )
        # sorted score from big to small
        kind_tp_and_score = []
        gt_num = {
            key: 0 for key in kinds_name
        }

        gt_has_used = []
        for gt_ in gt_kind_name_pos_score:
            gt_kind_name, _, _ = gt_
            gt_num[gt_kind_name] += 1
            gt_has_used.append(False)

        for pre_ in pre_kind_name_pos_score:
            pre_kind_name, pre_pos, pre_score = pre_
            is_tp = 0  # second element represents it tp(or fp)
            for gt_index, gt_ in enumerate(gt_kind_name_pos_score):
                gt_kind_name, gt_pos, gt_score = gt_
                if gt_kind_name == pre_kind_name and not gt_has_used[gt_index]:
                    iou = BaseTools.compute_iou(
                        list(pre_pos),
                        list(gt_pos)
                    )
                    if iou[0].item() > iou_th:
                        gt_has_used[gt_index] = True
                        is_tp = 1

            kind_tp_and_score.append(
                [pre_kind_name, is_tp, pre_score]
            )
        return kind_tp_and_score, gt_num

    @staticmethod
    def visualize(
            img: Union[torch.Tensor, np.ndarray],
            predict_name_pos_score: list,
            saved_path: str,
            class_colors: list,
            kinds_name: list,
    ):
        '''

        Args:
            img: just one image
            predict_name_pos_score: [kps0, kps1, ...]
            saved_path:
            class_colors: [color0, color1, ...]
            kinds_name: [kind_name0, kind_name1, ...]

        Returns:

        '''
        assert len(img.shape) == 3

        if not isinstance(img, np.ndarray):
            img = BaseTools.image_tensor_to_np(img)

        for box in predict_name_pos_score:
            if len(box) != 3:
                continue
            predict_kind_name, abs_double_pos, prob_score = box
            color = class_colors[kinds_name.index(predict_kind_name)]

            CV2.rectangle(img,
                          start_point=(int(abs_double_pos[0]), int(abs_double_pos[1])),
                          end_point=(int(abs_double_pos[2]), int(abs_double_pos[3])),
                          color=color,
                          thickness=2)

            # CV2.rectangle(img,
            #               start_point=(int(abs_double_pos[0]), int(abs_double_pos[1] - 20)),
            #               end_point=(int(abs_double_pos[2]), int(abs_double_pos[1])),
            #               color=color,
            #               thickness=-1)

            scale = 0.5
            CV2.putText(img,
                        '{}:{:.2%}'.format(predict_kind_name, prob_score),
                        org=(int(abs_double_pos[0]), int(abs_double_pos[1] - 5)),
                        font_scale=scale,
                        # color=tuple([255 - val for val in color]),
                        color=(0, 0, 0),
                        back_ground_color=color
                        )

        CV2.imwrite(saved_path, img)

    @staticmethod
    def iou(
            box_a: torch.Tensor,
            box_b: torch.Tensor
    ):
        # box_a/box_b (..., 4)
        box_a_x0y0 = box_a[..., :2]
        box_a_x1y1 = box_a[..., 2:]

        box_b_x0y0 = box_b[..., :2]
        box_b_x1y1 = box_b[..., 2:]

        s_a = torch.prod(box_a_x1y1 - box_a_x0y0, dim=-1)
        s_b = torch.prod(box_b_x1y1 - box_b_x0y0, dim=-1)

        inter_x0y0 = torch.max(box_a_x0y0, box_b_x0y0)
        inter_x1y1 = torch.min(box_a_x1y1, box_b_x1y1)

        inter_is_box = (inter_x1y1 > inter_x0y0).type(inter_x1y1.type()).prod(dim=-1)
        s_inter = torch.prod(inter_x1y1 - inter_x0y0, dim=-1) * inter_is_box
        union = s_a + s_b - s_inter
        iou = s_inter / (union + 1e-20)
        return torch.clamp(iou, 0, 1)

    @staticmethod
    def g_iou(
            box_a: torch.Tensor,
            box_b: torch.Tensor
    ):
        # box_a/box_b (..., 4)
        box_a_x0y0 = box_a[..., :2]
        box_a_x1y1 = box_a[..., 2:]

        box_b_x0y0 = box_b[..., :2]
        box_b_x1y1 = box_b[..., 2:]

        s_a = torch.prod(box_a_x1y1 - box_a_x0y0, dim=-1)
        s_b = torch.prod(box_b_x1y1 - box_b_x0y0, dim=-1)

        inter_x0y0 = torch.max(box_a_x0y0, box_b_x0y0)
        inter_x1y1 = torch.min(box_a_x1y1, box_b_x1y1)
        inter_is_box = (inter_x1y1 > inter_x0y0).type(inter_x1y1.type()).prod(dim=-1)
        s_inter = torch.prod(inter_x1y1 - inter_x0y0, dim=-1) * inter_is_box

        union = s_a + s_b - s_inter
        iou = s_inter / (union + 1e-20)

        outer_x0y0 = torch.min(box_a_x0y0, box_b_x0y0)
        outer_x1y1 = torch.max(box_a_x1y1, box_b_x1y1)
        outer_is_box = (outer_x1y1 > outer_x0y0).type(outer_x1y1.type()).prod(dim=-1)
        s_outer = torch.prod(outer_x1y1 - outer_x0y0, dim=-1) * outer_is_box
        s_rate = (s_outer - union) / (s_outer + 1e-20)

        g_iou = iou - s_rate
        return torch.clamp(g_iou, -1, 1)

    @staticmethod
    def d_iou(
            box_a: torch.Tensor,
            box_b: torch.Tensor
    ):
        # box_a/box_b (..., 4)
        box_a_x0y0 = box_a[..., :2]
        box_a_x1y1 = box_a[..., 2:]

        box_b_x0y0 = box_b[..., :2]
        box_b_x1y1 = box_b[..., 2:]

        s_a = torch.prod(box_a_x1y1 - box_a_x0y0, dim=-1)
        s_b = torch.prod(box_b_x1y1 - box_b_x0y0, dim=-1)

        inter_x0y0 = torch.max(box_a_x0y0, box_b_x0y0)
        inter_x1y1 = torch.min(box_a_x1y1, box_b_x1y1)
        inter_is_box = (inter_x1y1 > inter_x0y0).type(inter_x1y1.type()).prod(dim=-1)
        s_inter = torch.prod(inter_x1y1 - inter_x0y0, dim=-1) * inter_is_box

        union = s_a + s_b - s_inter
        iou = s_inter / (union + 1e-20)

        outer_x0y0 = torch.min(box_a_x0y0, box_b_x0y0)
        outer_x1y1 = torch.max(box_a_x1y1, box_b_x1y1)

        box_a_center = (box_a_x0y0 + box_a_x1y1) * 0.5
        box_b_center = (box_b_x0y0 + box_b_x1y1) * 0.5
        distance_center = torch.sum((box_b_center - box_a_center) ** 2, dim=-1)
        distance_outer = torch.sum((outer_x1y1 - outer_x0y0) ** 2, dim=-1)
        distance_rate = distance_center / (distance_outer + 1e-20)

        d_iou = iou - distance_rate
        return torch.clamp(d_iou, -1, 1)

    @staticmethod
    def c_iou(
            box_a: torch.Tensor,
            box_b: torch.Tensor
    ):
        # box_a/box_b (..., 4)
        box_a_x0y0 = box_a[..., :2]
        box_a_x1y1 = box_a[..., 2:]

        box_b_x0y0 = box_b[..., :2]
        box_b_x1y1 = box_b[..., 2:]

        s_a = torch.prod(box_a_x1y1 - box_a_x0y0, dim=-1)
        s_b = torch.prod(box_b_x1y1 - box_b_x0y0, dim=-1)

        inter_x0y0 = torch.max(box_a_x0y0, box_b_x0y0)
        inter_x1y1 = torch.min(box_a_x1y1, box_b_x1y1)
        inter_is_box = (inter_x1y1 > inter_x0y0).type(inter_x1y1.type()).prod(dim=-1)
        s_inter = torch.prod(inter_x1y1 - inter_x0y0, dim=-1) * inter_is_box

        union = s_a + s_b - s_inter
        iou = s_inter / (union + 1e-20)

        outer_x0y0 = torch.min(box_a_x0y0, box_b_x0y0)
        outer_x1y1 = torch.max(box_a_x1y1, box_b_x1y1)

        box_a_center = (box_a_x0y0 + box_a_x1y1) * 0.5
        box_b_center = (box_b_x0y0 + box_b_x1y1) * 0.5
        distance_center = torch.sum((box_b_center - box_a_center) ** 2, dim=-1)
        distance_outer = torch.sum((outer_x1y1 - outer_x0y0) ** 2, dim=-1)
        distance_rate = distance_center / (distance_outer + 1e-20)

        box_a_wh = box_a_x1y1 - box_a_x0y0
        box_b_wh = box_b_x1y1 - box_b_x0y0
        w2 = box_b_wh[..., 0]
        h2 = box_b_wh[..., 1]
        w1 = box_a_wh[..., 0]
        h1 = box_a_wh[..., 1]

        with torch.no_grad():
            arctan = torch.atan(w2 / (h2 + 1e-20)) - torch.atan(w1 / (h1 + 1e-20))
            v = (4 / (np.pi ** 2)) * torch.pow((torch.atan(w2 / (h2 + 1e-20)) - torch.atan(w1 / (h1 + 1e-20))), 2)
            S = 1 - iou
            alpha = v / (S + v + 1e-20)
            w_temp = 2 * w1

        ar = (8 / (np.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
        c_iou = iou - (distance_rate + alpha * ar)
        return torch.clamp(c_iou, -1, 1)
