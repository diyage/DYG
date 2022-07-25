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
    TYPE = 1

    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_iou_for_build_target(anchor_boxes, gt_box):
        """计算先验框和真实框之间的IoU
        Input: \n
            anchor_boxes: [K, 4] \n
                gt_box: [1, 4] \n
        Output: \n
                    iou : [K,] \n
        """

        # anchor box :
        ab_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
        # 计算先验框的左上角点坐标和右下角点坐标
        ab_x1y1_x2y2[:, 0] = anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2  # xmin
        ab_x1y1_x2y2[:, 1] = anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2  # ymin
        ab_x1y1_x2y2[:, 2] = anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2  # xmax
        ab_x1y1_x2y2[:, 3] = anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2  # ymax
        w_ab, h_ab = anchor_boxes[:, 2], anchor_boxes[:, 3]

        # gt_box :
        # 我们将真实框扩展成[K, 4], 便于计算IoU.
        gt_box_expand = np.repeat(gt_box, len(anchor_boxes), axis=0)

        gb_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
        # 计算真实框的左上角点坐标和右下角点坐标
        gb_x1y1_x2y2[:, 0] = gt_box_expand[:, 0] - gt_box_expand[:, 2] / 2  # xmin
        gb_x1y1_x2y2[:, 1] = gt_box_expand[:, 1] - gt_box_expand[:, 3] / 2  # ymin
        gb_x1y1_x2y2[:, 2] = gt_box_expand[:, 0] + gt_box_expand[:, 2] / 2  # xmax
        gb_x1y1_x2y2[:, 3] = gt_box_expand[:, 1] + gt_box_expand[:, 3] / 2  # ymin
        w_gt, h_gt = gt_box_expand[:, 2], gt_box_expand[:, 3]

        # 计算IoU
        S_gt = w_gt * h_gt
        S_ab = w_ab * h_ab
        I_w = np.minimum(gb_x1y1_x2y2[:, 2], ab_x1y1_x2y2[:, 2]) - np.maximum(gb_x1y1_x2y2[:, 0], ab_x1y1_x2y2[:, 0])
        I_h = np.minimum(gb_x1y1_x2y2[:, 3], ab_x1y1_x2y2[:, 3]) - np.maximum(gb_x1y1_x2y2[:, 1], ab_x1y1_x2y2[:, 1])
        S_I = I_h * I_w
        U = S_gt + S_ab - S_I + 1e-20
        IoU = S_I / U

        return IoU

    @staticmethod
    def set_anchors(anchor_size):
        """将输入进来的只包含wh的先验框尺寸转换成[N, 4]的ndarray类型，
           包含先验框的中心点坐标和宽高wh，中心点坐标设为0. \n
        Input: \n
            anchor_size: list -> [[h_1, w_1],  \n
                                  [h_2, w_2],  \n
                                   ...,  \n
                                  [h_n, w_n]]. \n
        Output: \n
            anchor_boxes: ndarray -> [[0, 0, anchor_w, anchor_h], \n
                                      [0, 0, anchor_w, anchor_h], \n
                                      ... \n
                                      [0, 0, anchor_w, anchor_h]]. \n
        """
        anchor_number = len(anchor_size)
        anchor_boxes = np.zeros([anchor_number, 4])
        for index, size in enumerate(anchor_size):
            anchor_w, anchor_h = size
            anchor_boxes[index] = np.array([0, 0, anchor_w, anchor_h])

        return anchor_boxes

    @staticmethod
    def generate_txtytwth(gt_label, w, h, s, anchor_size, iou_th):
        xmin, ymin, xmax, ymax = gt_label[:-1]
        # 计算真实边界框的中心点和宽高
        c_x = (xmax + xmin) / 2 * w
        c_y = (ymax + ymin) / 2 * h
        box_w = (xmax - xmin) * w
        box_h = (ymax - ymin) * h

        if box_w < 1e-4 or box_h < 1e-4:
            # print('not a valid data !!!')
            return False

            # 将真是边界框的尺寸映射到网格的尺度上去
        c_x_s = c_x / s
        c_y_s = c_y / s
        box_ws = box_w / s
        box_hs = box_h / s

        # 计算中心点所落在的网格的坐标
        grid_x = int(c_x_s)
        grid_y = int(c_y_s)

        # 获得先验框的中心点坐标和宽高，
        # 这里，我们设置所有的先验框的中心点坐标为0
        anchor_boxes = YOLOV2Tools.set_anchors(anchor_size)
        gt_box = np.array([[0, 0, box_ws, box_hs]])

        # 计算先验框和真实框之间的IoU
        iou = YOLOV2Tools.compute_iou_for_build_target(anchor_boxes, gt_box)

        # 只保留大于ignore_thresh的先验框去做正样本匹配,
        iou_mask = (iou > iou_th)

        result = []
        if iou_mask.sum() == 0:
            # 如果所有的先验框算出的IoU都小于阈值，那么就将IoU最大的那个先验框分配给正样本.
            # 其他的先验框统统视为负样本
            index = np.argmax(iou)
            p_w, p_h = anchor_size[index]
            tx = c_x_s - grid_x
            ty = c_y_s - grid_y
            tw = np.log(box_ws / p_w)
            th = np.log(box_hs / p_h)

            weight = 2.0 - (box_w / w) * (box_h / h)

            result.append([index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax])

            return result

        else:
            # 有至少一个先验框的IoU超过了阈值.
            # 但我们只保留超过阈值的那些先验框中IoU最大的，其他的先验框忽略掉，不参与loss计算。
            # 而小于阈值的先验框统统视为负样本。
            best_index = np.argmax(iou)
            for index, iou_m in enumerate(iou_mask):
                if iou_m:
                    if index == best_index:
                        p_w, p_h = anchor_size[index]
                        tx = c_x_s - grid_x
                        ty = c_y_s - grid_y
                        tw = np.log(box_ws / p_w)
                        th = np.log(box_hs / p_h)
                        weight = 2.0 - (box_w / w) * (box_h / h)

                        result.append([index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax])
                    else:
                        # 对于被忽略的先验框，我们将其权重weight设置为-1
                        result.append([index, grid_x, grid_y, 0., 0., 0., 0., -1.0, 0., 0., 0., 0.])

            return result

    @staticmethod
    def gt_creator(input_size, stride, label_lists, anchor_size, iou_th):
        # 必要的参数
        batch_size = len(label_lists)
        s = stride
        w = input_size
        h = input_size
        ws = w // s
        hs = h // s
        anchor_number = len(anchor_size)
        gt_tensor = np.zeros([batch_size, hs, ws, anchor_number, 1 + 1 + 4 + 1 + 4])

        # 制作正样本
        for batch_index in range(batch_size):
            for gt_label in label_lists[batch_index]:
                # get a bbox coords
                gt_class = int(gt_label[-1])
                results = YOLOV2Tools.generate_txtytwth(gt_label, w, h, s, anchor_size, iou_th)

                if results:
                    for result in results:
                        index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax = result

                        if weight > 0.:
                            if grid_y < gt_tensor.shape[1] and grid_x < gt_tensor.shape[2]:
                                gt_tensor[batch_index, grid_y, grid_x, index, 0] = 1.0
                                gt_tensor[batch_index, grid_y, grid_x, index, 1] = gt_class
                                gt_tensor[batch_index, grid_y, grid_x, index, 2:6] = np.array([tx, ty, tw, th])
                                gt_tensor[batch_index, grid_y, grid_x, index, 6] = weight
                                gt_tensor[batch_index, grid_y, grid_x, index, 7:] = np.array([xmin, ymin, xmax, ymax])
                        else:
                            # 对于那些被忽略的先验框，其gt_obj参数为-1，weight权重也是-1
                            gt_tensor[batch_index, grid_y, grid_x, index, 0] = -1.0
                            gt_tensor[batch_index, grid_y, grid_x, index, 6] = -1.0

        gt_tensor = gt_tensor.reshape(batch_size, hs * ws * anchor_number, -1)

        return gt_tensor

    @staticmethod
    def make_targets_1(
            labels: list,
            anchor_pre_wh: tuple,
            image_wh: tuple,
            grid_number: tuple,
            kinds_name: list,
            iou_th: float = 0.6,
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
            iou_th:
        Returns:

        '''

        label_lists = []
        for label in labels:
            label_lists.append(
                [
                    [obj[1] / image_wh[0], obj[2] / image_wh[1], obj[3] / image_wh[0], obj[4] / image_wh[1],
                     kinds_name.index(obj[0])] for obj in label
                ]

            )
        gt_np = YOLOV2Tools.gt_creator(
            image_wh[0],
            stride=image_wh[0] // grid_number[0],
            label_lists=label_lists,
            anchor_size=anchor_pre_wh,
            iou_th=iou_th,
        )
        gt_np = gt_np.reshape(len(labels), grid_number[1], grid_number[0], len(anchor_pre_wh), -1)
        gt_torch = torch.tensor(
            gt_np, dtype=torch.float32
        )
        # (N, H, W, a_n, 11)
        return gt_torch

    @staticmethod
    def make_targets_0(
            labels: list,
            anchor_pre_wh: tuple,
            image_wh: tuple,
            grid_number: tuple,
            kinds_name: list,
            iou_th: float = 0.6,
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
            iou_th:

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
                    image_wh,
                    iou_th
                )
                if best_index == -1:
                    continue

                grid_size = (
                    image_wh[0] // grid_number[0],
                    image_wh[1] // grid_number[1]
                )

                grid_index = (
                    int((abs_pos[0] + abs_pos[2]) * 0.5 // grid_size[0]),  # w -- on x-axis
                    int((abs_pos[1] + abs_pos[3]) * 0.5 // grid_size[1])  # h -- on y-axis
                )
                pos = tuple(abs_pos)

                for weight_index, weight_value in enumerate(weight_vec):
                    targets[batch_index, weight_index, 4, grid_index[1], grid_index[0]] = weight_value
                    # conf / weight --->
                    # -1, ignore
                    # 0, negative
                    # >0 [1, 2], positive
                    if weight_index == best_index:
                        targets[batch_index, weight_index, 0:4, grid_index[1], grid_index[0]] = torch.tensor(
                            pos)
                        targets[batch_index, weight_index, int(5 + kind_int), grid_index[1], grid_index[0]] = 1.0

        return targets.view(N, -1, H, W)

    @staticmethod
    def make_targets(
            labels: list,
            anchor_pre_wh: tuple,
            image_wh: tuple,
            grid_number: tuple,
            kinds_name: list,
            iou_th: float = 0.6,
    ) -> torch.Tensor:
        func_vec = [
            YOLOV2Tools.make_targets_0,
            YOLOV2Tools.make_targets_1,
        ]
        return func_vec[YOLOV2Tools.TYPE](
            labels,
            anchor_pre_wh,
            image_wh,
            grid_number,
            kinds_name,
            iou_th
        )

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
        iou_vec = []
        gt_w = abs_gt_pos[2] - abs_gt_pos[0]
        gt_h = abs_gt_pos[3] - abs_gt_pos[1]

        if gt_w < 1e-4 or gt_h < 1e-4:
            # valid obj box
            return -1, []

        s1 = gt_w * gt_h
        for index, val in enumerate(anchor_pre_wh):
            anchor_w = val[0] / grid_number[0] * image_wh[0]
            anchor_h = val[1] / grid_number[1] * image_wh[1]

            s0 = anchor_w * anchor_h
            inter = min(anchor_w, gt_w) * min(anchor_h, gt_h)
            union = s0 + s1 - inter
            iou = inter / (union + 1e-8)
            if iou >= best_iou:
                best_index = index
                best_iou = iou
            weight_vec.append(
                2.0 - (gt_w / image_wh[0]) * (gt_h / image_wh[1])
            )
            iou_vec.append(iou)

        for iou_index in range(len(iou_vec)):
            if iou_index != best_index:
                if iou_vec[iou_index] >= iou_th:
                    weight_vec[iou_index] = - 1.0  # ignore this anchor
                else:
                    weight_vec[iou_index] = 0.0  # negative anchor

        return best_index, weight_vec

    @staticmethod
    def split_target_0(
            x: torch.Tensor,
            anchor_number,
            *args,
            **kwargs
    ):
        N, C, H, W = x.shape
        K = C // anchor_number  # K = (x, y, w, h, conf, kinds0, kinds1, ...)
        # C = anchor_number * K
        x = x.view(N, anchor_number, K, H, W)
        x = x.permute(0, 3, 4, 1, 2)  # N * H * W * a_n * K

        position = [None, x[..., 0:4]]

        conf = x[..., 4]  # N * H * W * a_n
        cls_prob = x[..., 5:]  # N * H * W * a_n * ...

        res = {
            'position': position,  # first txty_(s)_twth, second xyxy(not scaled)
            'conf': conf,
            'cls_prob': cls_prob
        }
        return res

    @staticmethod
    def split_target_1(
            x: torch.Tensor,
            anchor_number,
            *args,
            **kwargs
    ):
        N, H, W, a_n, _ = x.shape
        conf = x[..., 0]
        cls_ind = x[..., 1]
        position = [x[..., 2:6], x[..., 7:]]
        # first position is txty_s_twth
        # second position is xyxy(scaled in (0, 1))
        weight = x[..., 6]
        res = {
            'position': position,
            'conf': conf,
            'cls_ind': cls_ind,
            'weight': weight
        }
        return res

    @staticmethod
    def split_target(
            x: torch.Tensor,
            anchor_number,
            *args,
            **kwargs
    ) -> dict:
        func_vec = [
            YOLOV2Tools.split_target_0,
            YOLOV2Tools.split_target_1,
        ]
        return func_vec[YOLOV2Tools.TYPE](
            x,
            anchor_number,
            *args,
            **kwargs
        )

    @staticmethod
    def split_model_out_0(
            x: torch.Tensor,
            anchor_number,
            *args,
            **kwargs
    ):
        N, C, H, W = x.shape
        K = C // anchor_number  # K = (x, y, w, h, conf, kinds0, kinds1, ...)
        # C = anchor_number * K
        x = x.view(N, anchor_number, K, H, W)
        x = x.permute(0, 3, 4, 1, 2)  # N * H * W * a_n * K

        position = [x[..., 0:4], None]

        conf = x[..., 4]  # N * H * W * a_n
        cls_prob = x[..., 5:]  # N * H * W * a_n * ...

        res = {
            'position': position,  # first txty_(s)_twth, second xyxy(not scaled)
            'conf': conf,
            'cls_prob': cls_prob
        }
        return res

    @staticmethod
    def split_model_out_1(
            x: torch.Tensor,
            anchor_number,
            *args,
            **kwargs
    ):
        if 'kinds_number' in kwargs.keys():
            kinds_number = kwargs['kinds_number']
        else:
            kinds_number = 20

        # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
        B, abC, H, W = x.shape
        prediction = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, abC)

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测
        # [B, H*W*num_anchor, 1]
        conf_pred = prediction[:, :, :1 * anchor_number].contiguous().view(B, H, W, anchor_number)
        # [B, H*W, num_anchor, num_cls]
        cls_pred = prediction[:, :,
                   1 * anchor_number: (1 + kinds_number) * anchor_number].contiguous().view(
            B, H, W, anchor_number, kinds_number)
        # [B, H*W, num_anchor, 4]
        txtytwth_pred = prediction[:, :, (1 + kinds_number) * anchor_number:].contiguous().view(
            B, H, W, anchor_number, 4
        )

        res = {
            'position': [txtytwth_pred, None],
            'conf': conf_pred,
            'cls_prob': cls_pred
        }
        return res

    @staticmethod
    def split_model_out(
            x: torch.Tensor,
            anchor_number,
            *args,
            **kwargs
    ) -> dict:
        func_vec = [
            YOLOV2Tools.split_model_out_0,
            YOLOV2Tools.split_model_out_1,
        ]
        return func_vec[YOLOV2Tools.TYPE](
            x,
            anchor_number,
            *args,
            **kwargs
        )

    @staticmethod
    def split_output(
            x: torch.Tensor,
            anchor_number,
            is_target: bool = False,
            *args,
            **kwargs
    ) -> dict:
        if YOLOV2Tools.TYPE == 1:
            return YOLOV2Tools.split_output_1(
                x,
                anchor_number,
                is_target,
                *args,
                **kwargs
            )
        else:
            return YOLOV2Tools.split_output_0(
                x,
                anchor_number,
                is_target,
                *args,
                **kwargs
            )

    @staticmethod
    def split_output_1(
            x: torch.Tensor,
            anchor_number,
            is_target: bool = False,
            *args,
            **kwargs
    ) -> dict:
        kinds_number = 20
        if not is_target:
            # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
            B, abC, H, W = x.shape
            prediction = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, abC)

            # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测
            # [B, H*W*num_anchor, 1]
            conf_pred = prediction[:, :, :1 * anchor_number].contiguous().view(B, H, W, anchor_number)
            # [B, H*W, num_anchor, num_cls]
            cls_pred = prediction[:, :,
                       1 * anchor_number: (1 + kinds_number) * anchor_number].contiguous().view(
                B, H, W, anchor_number, kinds_number)
            # [B, H*W, num_anchor, 4]
            txtytwth_pred = prediction[:, :, (1 + kinds_number) * anchor_number:].contiguous().view(
                B, H, W, anchor_number, 4
            )

            res = {
                'position': [txtytwth_pred, None],
                'conf': conf_pred,
                'cls_prob': cls_pred
            }

        else:
            N, H, W, a_n, _ = x.shape
            conf = x[..., 0]
            cls_ind = x[..., 1]
            position = [x[..., 2:6], x[..., 7:]]
            # first position is txty_s_twth
            # second position is xyxy(scaled in (0, 1))
            weight = x[..., 6]
            res = {
                'position': position,
                'conf': conf,
                'cls_ind': cls_ind,
                'weight': weight
            }

        return res

    @staticmethod
    def split_output_0(
            x: torch.Tensor,
            anchor_number,
            is_target: bool = False,
            *args,
            **kwargs
    ) -> dict:
        N, C, H, W = x.shape
        K = C // anchor_number   # K = (x, y, w, h, conf, kinds0, kinds1, ...)
        # C = anchor_number * K
        x = x.view(N, anchor_number, K, H, W)
        x = x.permute(0, 3, 4, 1, 2)  # N * H * W * a_n * K
        if is_target:
            position = [None, x[..., 0:4]]
        else:
            position = [x[..., 0:4], None]
        conf = x[..., 4]  # N * H * W * a_n
        cls_prob = x[..., 5:]  # N * H * W * a_n * ...

        res = {
            'position': position,  # first txty_(s)_twth, second xyxy(not scaled)
            'conf': conf,
            'cls_prob': cls_prob
        }
        return res

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
        return res
        # return res.clamp_(0.0, image_wh[0]-1)

    @staticmethod
    def xyxy_to_xywh(
            position: torch.Tensor,
            anchor_pre_wh: tuple,
            image_wh: tuple,
            grid_number: tuple,
    ) -> torch.Tensor:

        # def arc_sigmoid(x: torch.Tensor) -> torch.Tensor:
        #     return - torch.log(1.0 / (x + 1e-8) - 1.0)

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
        txy_s.clamp_(0.0, 1.0)  # be careful!!!, many center_x_y is zero !!!!
        twh = torch.log(w_h/image_wh[0]*grid_number[0]/pre_wh.expand_as(w_h) + 1e-20)

        return torch.cat((txy_s, twh), dim=-1)
