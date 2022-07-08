
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
