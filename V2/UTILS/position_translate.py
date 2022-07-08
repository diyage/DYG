import numpy as np


class Position:
    def __init__(self, p: tuple):
        self.a = None
        self.b = None
        self.m = None
        self.n = None
        self.set_position(p)

    def set_position(self, p: tuple):
        self.a = p[0]
        self.b = p[1]
        self.m = p[2]
        self.n = p[3]

    def get_position(self) -> tuple:
        return self.a, self.b, self.m, self.n

    def print(self):
        print("a: {}, b: {}, m: {}, n: {}".format(self.a, self.b, self.m, self.n))


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
                            a, b have not been processed by tanh --> [-1, 1]
                            (a little different from original one --> sigmoid)

        :param pre_box_w_h: (w, h)  k-means compute w and h on grid [0, grid_number[0] or grid_number[1]]
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
        self.grid_size = (self.image_size[0]/grid_number[0], self.image_size[1]/grid_number[1])

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

            a_ = np.arctanh(abs_center_x/self.image_size[0] * self.grid_number[0] - grid_index_to_x_axis)
            b_ = np.arctanh(abs_center_y/self.image_size[1] * self.grid_number[1] - grid_index_to_y_axis)
            m_ = np.log(obj_w/self.image_size[0]*self.grid_number[0]/self.pre_box_w_h[0])
            n_ = np.log(obj_h/self.image_size[1]*self.grid_number[1]/self.pre_box_w_h[1])

            self.center_offset_position = Position((a_, b_, m_, n_))

        elif types == 'center_offset':
            assert self.grid_index_to_x_y_axis is not None

            self.center_offset_position = Position(p)
            a, b, m, n = self.center_offset_position.get_position()

            grid_index_to_x_axis = self.grid_index_to_x_y_axis[0]
            grid_index_to_y_axis = self.grid_index_to_x_y_axis[1]

            abs_center_x = (np.tanh(a) + grid_index_to_x_axis)/self.grid_number[0]*self.image_size[0]
            abs_center_y = (np.tanh(b) + grid_index_to_y_axis)/self.grid_number[1]*self.image_size[1]
            obj_w = self.pre_box_w_h[0] * np.exp(m) /self.grid_number[0] * self.image_size[0]
            obj_h = self.pre_box_w_h[1] * np.exp(n) /self.grid_number[1] * self.image_size[1]

            a_ = max(abs_center_x - obj_w * 0.5, 0)
            b_ = max(abs_center_y - obj_h * 0.5, 0)
            m_ = min(abs_center_x + obj_w * 0.5, self.image_size[0])
            n_ = min(abs_center_y + obj_h * 0.5, self.image_size[1])
            self.abs_double_position = Position((a_, b_, m_, n_))
        else:
            print('wrong types={}'.format(types))
