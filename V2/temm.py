#
# from Tool import CV2, XMLTranslate
# from V2.Tool.position_translate import Position, PositionTranslate
#
#
# x = XMLTranslate('E:/yolo/', '2007_000032.xml')
# x.resize()
# img = x.img
# for obj in x.get_objects():
#
#     pos = (obj[1], obj[2], obj[3], obj[4])
#     pos_trans = PositionTranslate(pos, 'abs_double', image_size=(448, 448), pre_box_w_h=(3.0, 5.0))
#
#     p1 = pos_trans.abs_double_position.get_position()
#     g_ = pos_trans.grid_index_to_x_y_axis
#
#     p2 = pos_trans.center_offset_position.get_position()
#     print(p1)
#     print(p2)
#
#     pos_trans = PositionTranslate(p2, 'center_offset', image_size=(448, 448), pre_box_w_h=(3.0, 5.0), grid_index=g_)
#     print(pos_trans.abs_double_position.get_position())
#     print(pos_trans.center_offset_position.get_position())
#
#     break
#
#
#

# import torch
# x = torch.tensor([0, 1, 2])
# y = torch.tensor([0, 1, 2])
# repeat_on_x, repeat_on_y = torch.meshgrid(x, y)
#
# tmp = torch.cat([repeat_on_x.unsqueeze(-1), repeat_on_y.unsqueeze(-1)], dim=-1)
# for i in range(3):
#     for j in range(3):
#         print(tmp[i, j])

import torch


def compute_iou(bbox1, bbox2):
    """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
    Args:
        bbox1: (Tensor) bounding bboxes, sized [N, 4].
        bbox2: (Tensor) bounding bboxes, sized [M, 4].
    Returns:
        (Tensor) IoU, sized [N, M].
    """
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
    union = area1 + area2 - inter
    iou = inter / union

    return iou


a = torch.rand(size=(3, 4))
b = torch.rand(size=(4, 4))
c = compute_iou(a, b)
print(c.shape)
# N * 13 * 13 * 125
print(a.contiguous().view(-1,))