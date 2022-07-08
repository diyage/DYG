from V1.tools import XMLTranslate, PositionTranslate
from tools import CV2


x = XMLTranslate('E:/yolo/', '2007_000032.xml')
x.resize()
img = x.img
for obj in x.get_objects():
    CV2.putText(x.img, text=obj[0], org=(int(obj[1]), int(obj[2]-5)), color=(0, 255, 0))
    CV2.rectangle(x.img,
                  start_point=(int(obj[1]), int(obj[2])),
                  end_point=(int(obj[3]), int(obj[4])),
                  color=(255, 0, 0),
                  thickness=1)
    pos = (obj[1], obj[2], obj[3], obj[4])
    pos_trans = PositionTranslate(pos, 'abs_double', image_size=(448, 448))

    CV2.circle(x.img,
               center=(int(pos_trans.center_scaled_position.a * 448), int(pos_trans.center_scaled_position.b*448)),
               radius=3,
               color=(0, 0, 255),
               thickness=3)

    grid_size = pos_trans.grid_size[0]
    for i in range(8):
        tmp = int(i * grid_size)
        CV2.line(x.img, (0, tmp), (448, tmp), color=(0, 0, 255), thickness=2)
        CV2.line(x.img, (tmp, 0), (tmp, 448), color=(0, 0, 255), thickness=2)


CV2.imshow('a', x.img)
CV2.waitKey(0)

