from Tool.BaseTools import Position, BaseTools
import torch


class PositionTranslate:
    def __init__(self,
                 p: tuple,
                 types: str,
                 image_size: tuple,
                 grid_index: tuple = None,
                 grid_number: tuple = (7, 7)
                 ):

        '''
        a, b are distances to x and y axis
        :param p: (a,b,m,n)
        :param types: 'abs_double', 'center_scaled', 'center_offset'
        ‘abs_double’：     a,b --->(x1, y1)
                            m,n  ---> (x2, y2)

        ‘center_scaled’:   a,b ---> (center_img_scaled_x, center_img_scaled_y),
                            m,n ---> (scaled_img_width, scaled_img_height)

        'center_offset':   a,b ---> (center_grid_scaled_x, center_grid_scaled_y),
                            m,n ---> (scaled_img_width, scaled_img_height)

        :param image_size: (w, h)
        :param grid_index: (w, h)
        :param grid_number: (w, h)

        '''

        self.image_size = image_size

        self.center_scaled_position = None  # type: Position
        self.abs_double_position = None  # type: Position
        self.center_offset_position = None  # type: Position

        self.grid_index_to_x_y_axis = grid_index  # type:tuple
        self.grid_number = grid_number  # type:tuple
        self.grid_size = (self.image_size[0]/grid_number[0], self.image_size[1]/grid_number[1])

        if types == 'abs_double':
            self.abs_double_position = Position(p)
            self.center_scaled_position = self.abs_double_position_to_center_scaled_position(self.abs_double_position,
                                                                                             self.image_size)
            self.set_offset_position()

        elif types == 'center_scaled':
            self.center_scaled_position = Position(p)
            self.abs_double_position = self.center_scaled_position_to_abs_double_position(self.center_scaled_position,
                                                                                          self.image_size)
            self.set_offset_position()
        elif types == 'center_offset':
            self.center_offset_position = Position(p)
            grid_index_to_x_axis = self.grid_index_to_x_y_axis[0]
            grid_index_to_y_axis = self.grid_index_to_x_y_axis[1]

            a = self.grid_size[0] * (grid_index_to_x_axis + self.center_offset_position.a) / self.image_size[0]
            b = self.grid_size[1] * (grid_index_to_y_axis + self.center_offset_position.b) / self.image_size[1]
            m = self.center_offset_position.m
            n = self.center_offset_position.n

            self.center_scaled_position = Position(p=(a, b, m, n))
            self.abs_double_position = self.center_scaled_position_to_abs_double_position(self.center_scaled_position,
                                                                                          self.image_size)
        else:
            print('wrong types={}'.format(types))

    def set_offset_position(self):

        abs_center_a = self.center_scaled_position.a * self.image_size[0]
        abs_center_b = self.center_scaled_position.b * self.image_size[1]
        grid_index_to_x_axis = int(abs_center_a // self.grid_size[0])
        grid_index_to_y_axis = int(abs_center_b // self.grid_size[1])

        a = (abs_center_a - grid_index_to_x_axis * self.grid_size[0]) / self.grid_size[0]
        b = (abs_center_b - grid_index_to_y_axis * self.grid_size[1]) / self.grid_size[1]
        m = self.center_scaled_position.m
        n = self.center_scaled_position.n
        self.grid_index_to_x_y_axis = (grid_index_to_x_axis, grid_index_to_y_axis)  # be careful
        self.center_offset_position = Position(p=(a, b, m, n))

    @staticmethod
    def abs_double_position_to_center_scaled_position(abs_position: Position, image_size: tuple) -> Position:
        scaled_a = (1.0 * abs_position.a + 1.0 * abs_position.m) * 0.5 / image_size[0]  # point to x_axis_dist
        scaled_b = (1.0 * abs_position.b + 1.0 * abs_position.n) * 0.5 / image_size[1]  # point to y_axis_dist
        scaled_m = (1.0 * abs_position.m - 1.0 * abs_position.a) / image_size[0]
        scaled_n = (1.0 * abs_position.n - 1.0 * abs_position.b) / image_size[1]
        return Position(p=(scaled_a, scaled_b, scaled_m, scaled_n))

    @staticmethod
    def center_scaled_position_to_abs_double_position(scaled_position: Position, image_size: tuple) -> Position:
        img_w = scaled_position.m * image_size[0]
        img_h = scaled_position.n * image_size[1]

        abs_a = scaled_position.a * image_size[0] - 0.5 * img_w
        abs_a = max(abs_a, 0.0)

        abs_b = scaled_position.b * image_size[1] - 0.5 * img_h
        abs_b = max(abs_b, 0.0)

        abs_m = scaled_position.a * image_size[0] + 0.5 * img_w
        abs_m = min(abs_m, 1.0 * image_size[0] - 1)

        abs_n = scaled_position.b * image_size[1] + 0.5 * img_h
        abs_n = min(abs_n, 1.0 * image_size[1] - 1)

        return Position(p=(abs_a, abs_b, abs_m, abs_n))

    def print(self):
        print('grid_index_to_x_y_axis:')
        print(self.grid_index_to_x_y_axis)
        print('abs_double:')
        self.abs_double_position.print()
        print('center_scaled:')
        self.center_scaled_position.print()
        print('center_offset:')
        self.center_offset_position.print()


class YOLOV1Tools(BaseTools):
    def __init__(self):
        super().__init__()

    @staticmethod
    def make_targets(
            labels: list,
            image_wh: tuple,
            grid_number: tuple,
            kinds_name: list,
            need_abs: bool = False
    ):
        kinds_number = len(kinds_name)
        N, C, H, W = len(labels), 10 + kinds_number, grid_number[1], grid_number[0]

        target = torch.zeros(size=(N, C, H, W))

        # (x, y, w, h, c4) (x, y, w, h, c9) (class1, class2, ...)
        for batch_index, label in enumerate(labels):  # an image label
            for obj_index, obj in enumerate(label):  # many objects

                kind_int = kinds_name.index(obj[0])
                abs_pos = obj[1:]

                pos_trans = PositionTranslate(
                    abs_pos,
                    types='abs_double',
                    image_size=image_wh,
                    grid_number=grid_number
                    )

                if need_abs:
                    pos = pos_trans.abs_double_position.get_position()  # type: tuple
                else:
                    pos = pos_trans.center_offset_position.get_position()  # type: tuple

                grid_index = pos_trans.grid_index_to_x_y_axis  # type: tuple

                target[batch_index, 0:4, grid_index[1], grid_index[0]] = torch.tensor(pos)
                target[batch_index, 5:9, grid_index[1], grid_index[0]] = torch.tensor(pos)
                target[batch_index, [4, 9], grid_index[1], grid_index[0]] = 1
                target[batch_index, int(10 + kind_int), grid_index[1], grid_index[0]] = 1

        return target

    @staticmethod
    def split_output(
            x: torch.Tensor,
    ):

        x = x.permute(0, 2, 3, 1)  # (N, H, W, C)

        position_0 = x[..., 0:4]  # (N, H, W, 4)
        position_1 = x[..., 5:9]  # (N, H, W, 4)
        position_0 = position_0.unsqueeze(dim=-2)   # (N, H, W, 1, 4)
        position_1 = position_1.unsqueeze(dim=-2)  # (N, H, W, 1, 4)
        position = torch.cat((position_0, position_1), dim=-2)  # (N, H, W, 2, 4)

        conf = x[..., [4, 9]]  # N * H * W * 2
        cls_prob = x[..., 10:]  # (N, H, W, ...)
        cls_prob = cls_prob.unsqueeze(dim=-2)  # (N, H, W, 1, ...)
        cls_prob = torch.cat((cls_prob, cls_prob), dim=-2)  # (N, H, W, 2, ...)

        return position, conf, cls_prob

    @staticmethod
    def xywh_to_xyxy(
            position: torch.Tensor,
            image_wh: tuple,
            grid_number: tuple,
    ) -> torch.Tensor:
        # N * H * W * a_n * 4
        N, _, _, a_n, _ = position.shape

        grid = YOLOV1Tools.get_grid(grid_number)
        # H * W * 2

        grid = grid.unsqueeze(-2).unsqueeze(0).expand(N,
                                                      grid_number[0],
                                                      grid_number[1],
                                                      a_n,
                                                      2)

        # -1 * H * W * a_n * 2
        grid_index = grid.to(position.device)

        a_b = position[..., 0:2]  # -1 * a_n * 2
        m_n = position[..., 2:4]  # -1 * a_n * 2

        center_x_y = (a_b + grid_index) / grid_number[0] * image_wh[0]
        w_h = m_n * image_wh[0]

        x_y_0 = center_x_y - 0.5 * w_h
        x_y_1 = center_x_y + 0.5 * w_h
        res = torch.cat((x_y_0, x_y_1), dim=-1)
        return res.clamp_(0.0, image_wh[0])


