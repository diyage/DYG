
from UTILS import CV2, XMLTranslate
from V2.UTILS.position_translate import Position, PositionTranslate


x = XMLTranslate('E:/yolo/', '2007_000032.xml')
x.resize()
img = x.img
for obj in x.get_objects():

    pos = (obj[1], obj[2], obj[3], obj[4])
    pos_trans = PositionTranslate(pos, 'abs_double', image_size=(448, 448), pre_box_w_h=(3.0, 5.0))

    p1 = pos_trans.abs_double_position.get_position()
    g_ = pos_trans.grid_index_to_x_y_axis

    p2 = pos_trans.center_offset_position.get_position()
    print(p1)
    print(p2)

    pos_trans = PositionTranslate(p2, 'center_offset', image_size=(448, 448), pre_box_w_h=(3.0, 5.0), grid_index=g_)
    print(pos_trans.abs_double_position.get_position())
    print(pos_trans.center_offset_position.get_position())

    break



