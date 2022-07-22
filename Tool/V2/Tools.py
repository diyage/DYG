import torch
import numpy as np
from Tool.BaseTools import Position, BaseTools


class PositionTranslate:
    def __init__(self,
                 p: tuple,
                 types: str,
                 image_size: tuple,
                 pre_box_w_h: tuple,
                 grid_index: tuple = None,
                 grid_number: tuple = (13, 13)
                 ):

        '''
        a, b are distances to x and y axis
        :param p: (a,b,m,n)
        :param types: 'abs_double',  'center_offset'
        ‘abs_double’：     a,b --->(x1, y1)
                            m,n  ---> (x2, y2)

        'center_offset':   a,b ---> (center_grid_scaled_x, center_grid_scaled_y),
                            m,n ---> (scaled_img_width, scaled_img_height)
                            a, b have not been processed by sigmoid

        :param pre_box_w_h: (w, h)  k-means compute w and h
                            scaled on grid --> [0, grid_number[0] or grid_number[1]]
        :param image_size: (w, h) original image size
        :param grid_index: (w, h) which grid response(if types == 'center_offset' , offset where?)
        :param grid_number: (w, h)

        '''

        self.image_size = image_size
        self.pre_box_w_h = pre_box_w_h

        self.abs_double_position = None  # type: Position
        self.center_offset_position = None  # type: Position

        self.grid_index_to_x_y_axis = grid_index  # type:tuple
        self.grid_number = grid_number  # type:tuple
        self.grid_size = (
            self.image_size[0]/grid_number[0],
            self.image_size[1]/grid_number[1]
        )

        if types == 'abs_double':
            self.abs_double_position = Position(p)
            a, b, m, n = self.abs_double_position.get_position()

            abs_center_x = (a + m) * 0.5
            abs_center_y = (b + n) * 0.5
            obj_w = m - a
            obj_h = n - b

            grid_index_to_x_axis = int(abs_center_x // self.grid_size[0])
            grid_index_to_y_axis = int(abs_center_y // self.grid_size[1])
            self.grid_index_to_x_y_axis = (grid_index_to_x_axis, grid_index_to_y_axis)

            a_ = self.arc_sigmoid(abs_center_x/self.image_size[0] * self.grid_number[0] - grid_index_to_x_axis)
            b_ = self.arc_sigmoid(abs_center_y/self.image_size[1] * self.grid_number[1] - grid_index_to_y_axis)
            m_ = np.log(obj_w/self.image_size[0]*self.grid_number[0]/self.pre_box_w_h[0])
            n_ = np.log(obj_h/self.image_size[1]*self.grid_number[1]/self.pre_box_w_h[1])

            self.center_offset_position = Position((a_, b_, m_, n_))

        elif types == 'center_offset':
            assert self.grid_index_to_x_y_axis is not None

            self.center_offset_position = Position(p)
            a, b, m, n = self.center_offset_position.get_position()

            grid_index_to_x_axis = self.grid_index_to_x_y_axis[0]
            grid_index_to_y_axis = self.grid_index_to_x_y_axis[1]

            abs_center_x = (self.sigmoid(a) + grid_index_to_x_axis)/self.grid_number[0]*self.image_size[0]
            abs_center_y = (self.sigmoid(b) + grid_index_to_y_axis)/self.grid_number[1]*self.image_size[1]
            obj_w = self.pre_box_w_h[0] * np.exp(m) / self.grid_number[0] * self.image_size[0]
            obj_h = self.pre_box_w_h[1] * np.exp(n) / self.grid_number[1] * self.image_size[1]

            a_ = max(abs_center_x - obj_w * 0.5, 0)
            b_ = max(abs_center_y - obj_h * 0.5, 0)
            m_ = min(abs_center_x + obj_w * 0.5, self.image_size[0])
            n_ = min(abs_center_y + obj_h * 0.5, self.image_size[1])
            self.abs_double_position = Position((a_, b_, m_, n_))
        else:
            print('wrong types={}'.format(types))

    @staticmethod
    def sigmoid(x) -> np.ndarray:
        s = 1.0 / (1.0 + np.exp(-x))

        return s

    @staticmethod
    def arc_sigmoid(x) -> np.ndarray:
        return - np.log(1.0 / (x + 1e-8) - 1.0)


class YOLOV2Tools(BaseTools):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_anchor_response_result(
            anchor_pre_wh: tuple,
            abs_gt_pos: tuple,
            grid_number: tuple,
            image_wh: tuple,
            iou_th: float = 0.6,
    ):
        best_index = 0
        best_iou = 0
        weight_vec = []
        for index, val in enumerate(anchor_pre_wh):
            anchor_w = val[0] / grid_number[0] * image_wh[0]
            anchor_h = val[1] / grid_number[1] * image_wh[1]
            gt_w = abs_gt_pos[2] - abs_gt_pos[0]
            gt_h = abs_gt_pos[3] - abs_gt_pos[1]

            s0 = anchor_w * anchor_h
            s1 = gt_w * gt_h
            inter = min(anchor_w, gt_w) * min(anchor_h, gt_h)
            union = s0 + s1 - inter
            iou = inter/union
            if iou >= best_iou:
                best_index = index
                best_iou = iou
            weight_vec.append(
                2.0 - (gt_w / image_wh[0]) * (gt_h / image_wh[1])
            )
        for weight_index in range(len(weight_vec)):
            if weight_index != best_index:
                if weight_vec[weight_index] >= iou_th:
                    weight_vec[weight_index] = - 1.0  # ignore this anchor
                else:
                    weight_vec[weight_index] = 0.0  # negative anchor

        return best_index, weight_vec

    @staticmethod
    def make_targets(
            labels: list,
            anchor_pre_wh: tuple,
            image_wh: tuple,
            grid_number: tuple,
            kinds_name: list,
            need_abs: bool = False
    ) -> torch.Tensor:
        '''

        Args:
            labels: [
                [obj, obj, obj, ...],               --> one image
                ...
            ]
                obj = [kind_name: str, x, y, x, y]  --> one obj
            anchor_pre_wh: [
                [w0, h0],
                [w1, h1],
                ...
            ]
            image_wh: [image_w, image_h]
            grid_number: [grid_w, grid_h]
            kinds_name: [kind_name0, kinds_name1, ... ]
            need_abs: Position type, (x, y, x, y) or (tx, ty, tw, th) ?

        Returns:
            (N, a_n * (5 + kinds_number), H, W)
        '''

        kinds_number = len(kinds_name)
        N, a_n, H, W = len(labels), len(anchor_pre_wh), grid_number[1], grid_number[0]

        targets = torch.zeros(size=(N, a_n, 5 + kinds_number, H, W))

        for batch_index, label in enumerate(labels):  # an image label
            for obj_index, obj in enumerate(label):  # many objects
                kind_int = kinds_name.index(obj[0])
                abs_pos = obj[1:]

                best_index, weight_vec = YOLOV2Tools.compute_anchor_response_result(
                    anchor_pre_wh,
                    abs_pos,
                    grid_number,
                    image_wh
                )
                pos_trans = PositionTranslate(
                    abs_pos,
                    types='abs_double',
                    image_size=image_wh,
                    pre_box_w_h=anchor_pre_wh[best_index],
                    grid_number=grid_number
                )

                if need_abs:
                    pos = pos_trans.abs_double_position.get_position()  # type: tuple
                else:
                    pos = pos_trans.center_offset_position.get_position()  # type: tuple

                grid_index = pos_trans.grid_index_to_x_y_axis  # type: tuple

                for weight_index, weight_value in enumerate(weight_vec):
                    # targets[batch_index, weight_index, 0:4, grid_index[1], grid_index[0]] = torch.tensor(
                    #     pos)
                    #
                    # targets[batch_index, weight_index, 4, grid_index[1], grid_index[0]] = 1.0
                    # # conf / weight
                    #
                    # targets[batch_index, weight_index, int(5 + kind_int), grid_index[1], grid_index[0]] = 1.0

                    # just one box response
                    # weight index is also the anchor(box) index
                    # weight_val is -1, 0, or >0
                    targets[batch_index, weight_index, 0:4, grid_index[1], grid_index[0]] = torch.tensor(
                        pos)

                    targets[batch_index, weight_index, 4, grid_index[1], grid_index[0]] = weight_value
                    # conf / weight

                    targets[batch_index, weight_index, int(5 + kind_int), grid_index[1], grid_index[0]] = 1.0

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
        cls_prob = x[..., 5:]  # N * H * W * a_n * ...

        return position, conf, cls_prob

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
        w_h = (torch.exp(m_n) - 1e-8) * pre_wh.expand_as(m_n) / grid_number[0] * image_wh[0]

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

        def arc_sigmoid(x: torch.Tensor) -> torch.Tensor:
            return - torch.log(1.0 / (x + 1e-8) - 1.0)

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

        # txy = arc_sigmoid(center_x_y / image_wh[0] * grid_number[0] - grid_index)
        txy_s = center_x_y / image_wh[0] * grid_number[0] - grid_index
        # center_xy = (sigmoid(txy) + grid_index) / grid_number * image_wh
        # we define txy_s = sigmoid(txy)
        # be careful ,we do not use arc_sigmoid method
        # if you use txy(in model output), please make sure (use sigmoid)

        twh = torch.log(w_h/image_wh[0]*grid_number[0]/pre_wh.expand_as(w_h) + 1e-8)

        return torch.cat((txy_s, twh), dim=-1)
