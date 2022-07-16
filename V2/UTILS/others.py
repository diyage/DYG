import torch
import torch.nn as nn
from typing import Union
import numpy as np
from UTILS import CV2
from V2.UTILS.position_translate import PositionTranslate


def arc_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return - torch.log(1.0 / (x + 1e-8) - 1.0)


class YOLOV2Tools:

    @staticmethod
    def compute_iou_m_to_n(
            bbox1: Union[torch.Tensor, np.ndarray, list],
            bbox2: Union[torch.Tensor, np.ndarray, list]
    )-> torch.Tensor:

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
            bbox1[:, :2].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Conpute right-bottom coordinate of the intersections
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute area of the intersections from the coordinates
        wh = rb - lt   # width and height of the intersection, [N, M, 2]
        wh[wh < 0] = 0 # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

        # Compute area of the bboxes
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
        area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]

        # Compute IoU from the areas
        union = area1 + area2 - inter  # [N, M]
        iou = inter / union           # [N, M]
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

        w0 = boxes0[..., 2] - boxes0[..., 0]  # -1
        h0 = boxes0[..., 3] - boxes0[..., 1]  # -1
        s0 = w0 * h0  # -1

        w1 = boxes1[..., 2] - boxes1[..., 0]  # -1
        h1 = boxes1[..., 3] - boxes1[..., 1]  # -1
        s1 = w1 * h1  # -1

        boxes = torch.stack((boxes0, boxes1), dim=-1)  # # -1 * 4 * 2

        inter_boxes_a_b = torch.max(boxes[..., 0:2, :], dim=-1)[0]  # # -1 * 2

        inter_boxes_m_n = torch.min(boxes[..., 2:4, :], dim=-1)[0]  # # -1 * 2

        inter_boxes_w_h = inter_boxes_m_n - inter_boxes_a_b  # # -1 * 2
        inter_boxes_w_h[inter_boxes_w_h < 0] = 0.0  # # -1 * 2
        inter_boxes_s = inter_boxes_w_h[..., 0] * inter_boxes_w_h[..., 1]  # # -1

        union_boxes_s = s0 + s1 - inter_boxes_s
        return inter_boxes_s / union_boxes_s  # -1

    @staticmethod
    def make_targets(
            labels: list,
            anchor_pre_wh: tuple,
            image_wh: tuple,
            grid_number: tuple,
            kinds_name: list,
            need_abs: bool = False
    ) -> torch.Tensor:

        kinds_number = len(kinds_name)
        N, a_n, H, W = len(labels), len(anchor_pre_wh), grid_number[1], grid_number[0]

        targets = torch.zeros(size=(N, a_n, 5 + kinds_number, H, W))

        for batch_index, label in enumerate(labels):  # an image label
            for obj_index, obj in enumerate(label):  # many objects
                kind_int = kinds_name.index(obj[0])
                abs_pos = obj[1:]
                for pre_box_index, pre_box_w_h in enumerate(anchor_pre_wh):
                    pos_trans = PositionTranslate(abs_pos,
                                                  types='abs_double',
                                                  image_size=image_wh,
                                                  pre_box_w_h=pre_box_w_h,
                                                  grid_number=grid_number)

                    if need_abs:
                        pos = pos_trans.abs_double_position.get_position()  # type: tuple
                    else:
                        pos = pos_trans.center_offset_position.get_position()  # type: tuple

                    grid_index = pos_trans.grid_index_to_x_y_axis  # type: tuple

                    targets[batch_index, pre_box_index, 0:4, grid_index[1], grid_index[0]] = torch.tensor(
                        pos)

                    targets[batch_index, pre_box_index, 4, grid_index[1], grid_index[0]] = 1.0

                    targets[batch_index, pre_box_index, int(5 + kind_int):, grid_index[1], grid_index[0]] = 1.0

        return targets.view(N, -1, H, W)

    @staticmethod
    def split_output(
            x: torch.Tensor,
            anchor_number,
    ):

        N, C, H, W = x.shape
        K = C // anchor_number   # K = (x, y, w, h, conf, kinds0, kinds1, ...)
        # C = anchor_number * K
        x = x.view(N, anchor_number, K, H, W)
        x = x.permute(0, 3, 4, 1, 2)  # N * H * W * a_n * K

        position = x[..., 0:4]  # N * H * W * a_n * 4
        conf = x[..., 4]  # N * H * W * a_n
        scores = x[..., 5:]  # N * H * W * a_n * ...

        return position, conf, scores

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
    def xywh_to_xyxy(
            position: torch.Tensor,
            anchor_pre_wh: tuple,
            image_wh: tuple,
            grid_number: tuple,
    ) -> torch.Tensor:
        # -1 * H * W * a_n * 4
        N, _, _, a_n, _ = position.shape

        grid = YOLOV2Tools.get_grid(grid_number)
        # H * W * 2

        grid = grid.unsqueeze(-2).unsqueeze(0).expand(N,
                                                      grid_number[0],
                                                      grid_number[1],
                                                      a_n,
                                                      2)

        # -1 * H * W * a_n * 2

        grid_index = grid.to(position.device)

        pre_wh = torch.tensor(
            anchor_pre_wh,
            dtype=torch.float32
        )
        pre_wh = pre_wh.to(position.device)
        # a_n * 2

        a_b = position[..., 0:2]  # -1 * a_n * 2
        m_n = position[..., 2:4]  # -1 * a_n * 2

        center_x_y = (torch.sigmoid(a_b) + grid_index) / grid_number[0] * image_wh[0]
        w_h = torch.exp(m_n) * pre_wh.expand_as(m_n) / grid_number[0] * image_wh[0]

        x_y_0 = center_x_y - 0.5 * w_h
        # x_y_0[x_y_0 < 0] = 0
        x_y_1 = center_x_y + 0.5 * w_h
        # x_y_1[x_y_1 > grid_number] = grid_number
        res = torch.cat((x_y_0, x_y_1), dim=-1)
        return res.clamp_(0.0, image_wh[0])

    @staticmethod
    def xyxy_to_xywh(
            position: torch.Tensor,
            anchor_pre_wh: tuple,
            image_wh: tuple,
            grid_number: tuple,
    ) -> torch.Tensor:
        # -1 * H * W * a_n * 4
        N, _, _, a_n, _ = position.shape

        grid = YOLOV2Tools.get_grid(grid_number)
        # H * W * 2

        grid = grid.unsqueeze(-2).unsqueeze(0).expand(N,
                                                      grid_number[0],
                                                      grid_number[1],
                                                      a_n,
                                                      2)

        # -1 * H * W * a_n * 2

        grid_index = grid.to(position.device)

        pre_wh = torch.tensor(
            anchor_pre_wh,
            dtype=torch.float32
        )
        pre_wh = pre_wh.to(position.device)

        # a_n * 2

        a_b = position[..., 0:2]  # -1 * a_n * 2
        m_n = position[..., 2:4]  # -1 * a_n * 2

        center_x_y = (a_b + m_n) * 0.5

        w_h = m_n - a_b

        txy = arc_sigmoid(center_x_y / image_wh[0] * grid_number[0] - grid_index)

        twh = torch.log(w_h/image_wh[0]*grid_number[0]/pre_wh.expand_as(w_h))

        return torch.cat((txy, twh), dim=-1)

    @staticmethod
    def translate_to_abs_position(
            position: torch.Tensor,
            anchor_pre_wh: tuple,
            image_wh: tuple,
            grid_number: tuple,
    ) -> torch.Tensor:
        N, H, W, a_n, _ = position.shape
        position_abs = torch.zeros(size=position.shape).to(position.device)

        for i in range(N):
            for r in range(H):
                for c in range(W):
                    for a in range(a_n):
                        pos = position[i, r, c,  a, :]

                        p = (pos[0].item(), pos[1].item(), pos[2].item(), pos[3].item())
                        p_trans = PositionTranslate(
                            p,
                            types='center_offset',
                            image_size=image_wh,
                            pre_box_w_h=anchor_pre_wh[a],
                            grid_index=(c, r),
                            grid_number=grid_number
                        )
                        x0, y0, x1, y1 = p_trans.abs_double_position.get_position()
                        position_abs[i, r, c, a, 0] = x0
                        position_abs[i, r, c, a, 1] = y0
                        position_abs[i, r, c, a, 2] = x1
                        position_abs[i, r, c, a, 3] = y1

        return position_abs

    @staticmethod
    def nms(
            position_abs: torch.Tensor,
            conf: torch.Tensor,
            threshold: float = 0.5
    ):
        x1 = position_abs[:, 0]
        y1 = position_abs[:, 1]
        x2 = position_abs[:, 2]
        y2 = position_abs[:, 3]
        areas = (x2 - x1) * (y2 - y1)  # [N,]
        _, order = conf.sort(0, descending=True)

        keep = []
        while order.numel() > 0:
            if order.numel() == 1:  # just one
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()  # max conf
                keep.append(i)

            # 计算box[i]与其余各框的IOU(思路很好)
            xx1 = x1[order[1:]].clamp(min=x1[i])  # [N-1,]
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])
            inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)  # [N-1,]

            iou = inter / (areas[i] + areas[order[1:]] - inter)  # [N-1,]
            idx = (iou <= threshold).nonzero().squeeze()  # idx[N-1,] order[N,]
            if idx.numel() == 0:
                break
            order = order[idx + 1]  #

        return torch.tensor(keep, dtype=torch.long)

    @staticmethod
    def decode_out(
            out: torch.Tensor,
            anchor_pre_wh: tuple,
            image_wh: tuple,
            grid_number: tuple,
            kins_name: list,
            iou_th: float = 0.5,
            conf_th: float = 0.1,
            prob_th: float = 0.1,
            use_score: bool = True,
            use_conf: bool = False,
            out_is_target: bool = False,
    ) -> list:
        shape = out.shape
        assert len(shape) == 3
        _, H, W = shape
        out = out.unsqueeze(0)  # 1 * _ * H * N

        a_n = len(anchor_pre_wh)

        position, conf, scores = YOLOV2Tools.split_output(
            out,
            a_n,
        )
        if not out_is_target:
            conf = nn.Sigmoid()(conf)
            scores = nn.Softmax(dim=-1)(scores)

        # 1 * H * W * a_n * ()
        position_abs = YOLOV2Tools.translate_to_abs_position(
            position,
            anchor_pre_wh,
            image_wh,
            grid_number
        )

        # position_abs = YOLOV2Tools.xywh_to_xyxy(
        #     position,
        #     anchor_pre_wh,
        #     image_wh,
        #     grid_number
        # )

        position_abs_ = position_abs.contiguous().view(-1, 4)
        conf_ = conf.contiguous().view(-1,)
        scores_ = scores.contiguous().view(-1, len(kins_name))

        keep_index = YOLOV2Tools.nms(
            position_abs_,
            conf_,
            threshold=iou_th
        )

        res = []
        for index in keep_index:

            predict_value, predict_index = scores_[index].max(dim=-1)

            if conf_[index] > conf_th and predict_value > prob_th:
                abs_double_pos = tuple(position_abs_[index].cpu().detach().numpy().tolist())

                predict_kind_name = kins_name[int(predict_index.item())]
                prob_score = predict_value.item()

                tmp = [predict_kind_name, abs_double_pos]

                if use_score:
                    tmp.append(prob_score)

                if use_conf:
                    tmp.append(conf_[index].item())

                res.append(tuple(tmp))
        return res

    @staticmethod
    def visualize(
            img: Union[torch.Tensor, np.ndarray],
            predict_name_pos_score: list,
            saved_path: str
    ):
        assert len(img.shape) == 3

        if not isinstance(img, np.ndarray):

            img = (img.cpu().detach().numpy().copy()*0.5 + 0.5) * 255  # type:np.ndarray
            img = np.transpose(img, axes=(1, 2, 0))  # type:np.ndarray
            img = np.array(img, np.uint8)  # type:np.ndarray
            img = CV2.cvtColorToBGR(img)

        for box in predict_name_pos_score:
            predict_kind_name, abs_double_pos, prob_score = box
            color = (0, 0, 255)
            CV2.putText(img,
                        '{}:{:.2%}'.format(predict_kind_name, prob_score),
                        org=(int(abs_double_pos[0]), int(abs_double_pos[1] + 12)),
                        color=color)

            CV2.rectangle(img,
                          start_point=(int(abs_double_pos[0]), int(abs_double_pos[1])),
                          end_point=(int(abs_double_pos[2]), int(abs_double_pos[3])),
                          color=color,
                          thickness=2)

        CV2.imwrite(saved_path, img)

    @staticmethod
    def compute_AP(
            vec_pred_dict: dict,
            vec_target_dict: dict,
            iou_th: float
    )-> np.ndarray:

        def compute_cls_ap(cls: str):
            pred = vec_pred_dict[cls]  # type: list
            pred = sorted(pred, key=lambda s: s[-1], reverse=True)
            # sorted by conf
            # -1 * 3(image_index, pos, conf)
            target = vec_target_dict[cls]  # type: list

            tp = 0.0
            fp = 0.0

            total_box_number = len(target)

            if total_box_number == 0:
                return -1.0

            has_used = [False for _ in range(total_box_number)]

            precision_vec = []
            recall_vevc = []
            if len(pred) == 0:
                return 0.0

            for pre_image_index, pre_pos, _ in pred:
                for now_ind, (gt_image_index, gt_pos, _) in enumerate(target):

                    if pre_image_index != gt_image_index:
                        continue

                    iou = YOLOV2Tools.compute_iou(list(pre_pos), list(gt_pos))[0].item()
                    if iou > iou_th and not has_used[now_ind]:
                        tp += 1.0
                        has_used[now_ind] = True
                    else:
                        fp += 1.0

                    precision = tp / (tp + fp)
                    precision_vec.append(precision)
                    recall = tp / total_box_number
                    recall_vevc.append(recall)

            if len(recall_vevc) == 0:
                return -1.0

            precision_recall = list(zip(precision_vec, recall_vevc))
            precision_recall = sorted(precision_recall, key=lambda s: s[-1], reverse=False)
            precision_recall = np.array(precision_recall, dtype=np.float32)
            ap = np.trapz(precision_recall[:, 0], precision_recall[:, 1], dx=0.01)
            return ap

        ap_vec = []
        for kinds_name in vec_pred_dict.keys():
            ap = compute_cls_ap(kinds_name)
            ap_vec.append(ap)
        return np.array(ap_vec)

    @staticmethod
    def compute_mAP(
            pre_boxes: list,
            target_boxes: list,
            kinds_name: list,
            iou_th: float
    ):

        N = len(pre_boxes)
        # pre_boxes --> N * -1 * 3
        # 3 --> kind_name, position, conf

        vec_pred_dict = {
            key: [] for key in kinds_name
        }
        vec_target_dict = {
            key: [] for key in kinds_name
        }

        for image_index in range(N):
            for box in pre_boxes[image_index]:
                predict_kind_name, abs_double_pos, conf = box
                vec_pred_dict[predict_kind_name].append(
                    [image_index,
                     abs_double_pos,
                     conf]
                )
            for box in target_boxes[image_index]:
                gt_kind_name, gt_pos, conf = box
                vec_target_dict[gt_kind_name].append(
                    [image_index,
                     gt_pos,
                     conf]
                )

        # -1 * 4
        ap_vec = YOLOV2Tools.compute_AP(
            vec_pred_dict,
            vec_target_dict,
            iou_th
        )
        mAP = ap_vec[ap_vec >= 0].mean()
        return mAP







