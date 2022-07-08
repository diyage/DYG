import torch
import torch.nn as nn
import numpy as np
from V1.tools.position_translate import PositionTranslate
from UTILS import CV2


class YoLoV1Predictor:
    def __init__(self,
                 net: nn.Module,
                 conf_th: float = 0.1,
                 prob_th: float = 0.1,
                 iou_th: float = 0.5,
                 image_size: tuple = (448, 448),
                 grid_number: tuple = (7, 7),
                 kinds_name: list = None
                 ):
        # must be careful, this part is different from form YOLO
        # we have not use NMS...
        self.net = net
        self.conf_th = conf_th
        self.prob_th = prob_th  # conf * class_score
        self.iou_th = iou_th
        self.image_size = image_size
        self.grid_number = grid_number
        self.kinds_name = kinds_name

    def show(self, img: np.ndarray, boxes: list):
        img = img.copy()
        for box in boxes:
            predict_kind_name, abs_double_pos, prob_score = box
            color = (0, 0, 255)
            CV2.putText(img,
                        '{}:{:.2%}'.format(predict_kind_name, prob_score),
                        org=(int(abs_double_pos[0]), int(abs_double_pos[1] - 5)),
                        color=color)

            CV2.rectangle(img,
                          start_point=(int(abs_double_pos[0]), int(abs_double_pos[1])),
                          end_point=(int(abs_double_pos[2]), int(abs_double_pos[3])),
                          color=color,
                          thickness=1)

            pos = (abs_double_pos[0], abs_double_pos[1], abs_double_pos[2], abs_double_pos[3])

            pos_trans = PositionTranslate(pos, 'abs_double', image_size=self.image_size, grid_number=self.grid_number)
            CV2.circle(img,
                       center=(
                           int(pos_trans.center_scaled_position.a * self.image_size[0]),
                           int(pos_trans.center_scaled_position.b * self.image_size[1])),
                       radius=3,
                       color=(0, 0, 255),
                       thickness=3)

            grid_size = pos_trans.grid_size[0]
            for i in range(self.grid_number[0] + 1):
                tmp = int(i * grid_size)
                CV2.line(img, (0, tmp), (self.image_size[0], tmp), color=(0, 0, 255), thickness=2)
                CV2.line(img, (tmp, 0), (tmp, self.image_size[1]), color=(0, 0, 255), thickness=2)

        CV2.imshow('a', img)
        CV2.waitKey(0)

    def visualize(self, img: torch.Tensor, out: torch.Tensor, saved_path: str):
        assert len(img.shape) == 3
        img = img.cpu().detach().numpy().copy() * 255  # type:np.ndarray
        img = np.transpose(img, axes=(1, 2, 0))  # type:np.ndarray
        img = np.array(img, np.uint8)  # type:np.ndarray
        img = CV2.cvtColorToBGR(img)

        out = out.cpu().detach().numpy().copy()  # type:np.ndarray

        boxes = self.decode_out(out)

        for box in boxes:
            predict_kind_name, abs_double_pos, prob_score = box
            color = (0, 0, 255)
            CV2.putText(img,
                        '{}:{:.2%}'.format(predict_kind_name, prob_score),
                        org=(int(abs_double_pos[0]), int(abs_double_pos[1] - 5)),
                        color=color)

            CV2.rectangle(img,
                          start_point=(int(abs_double_pos[0]), int(abs_double_pos[1])),
                          end_point=(int(abs_double_pos[2]), int(abs_double_pos[3])),
                          color=color,
                          thickness=2)

            pos = (abs_double_pos[0], abs_double_pos[1], abs_double_pos[2], abs_double_pos[3])

            pos_trans = PositionTranslate(pos, 'abs_double', image_size=self.image_size, grid_number=self.grid_number)
            CV2.circle(img,
                       center=(
                           int(pos_trans.center_scaled_position.a * self.image_size[0]),
                           int(pos_trans.center_scaled_position.b * self.image_size[1])),
                       radius=3,
                       color=(0, 0, 255),
                       thickness=3)

            # grid_size = pos_trans.grid_size[0]
            # for i in range(self.grid_number[0] + 1):
            #     tmp = int(i * grid_size)
            #     CV2.line(img, (0, tmp), (self.image_size[0], tmp), color=(0, 0, 255), thickness=2)
            #     CV2.line(img, (tmp, 0), (tmp, self.image_size[1]), color=(0, 0, 255), thickness=2)
        CV2.imwrite(saved_path, img)

    def decode_out(self, y: np.ndarray) -> list:
        res = []
        for r in range(self.grid_number[1]):
            for c in range(self.grid_number[0]):
                # print('{},{}'.format(y[r, c, 4], y[r, c, 9]))
                if y[r, c, 4] < self.conf_th and y[r, c, 9] < self.conf_th:
                    continue
                else:
                    # print('has box')
                    pass

                class_scores = y[r, c, 10:]
                max_class_score_index = np.argmax(class_scores)

                if y[r, c, 4] > y[r, c, 9]:  # box1
                    box = 0
                else:  # box2
                    box = 1

                prob_score = y[r, c, box * 5 + 4] * class_scores[max_class_score_index]
                if prob_score < self.prob_th:
                    continue

                center_offset_pos = (
                    y[r, c, box * 5 + 0],
                    y[r, c, box * 5 + 1],
                    y[r, c, box * 5 + 2],
                    y[r, c, box * 5 + 3]
                )
                grid_index_to_x_y_axis = (c, r)

                p_trans = PositionTranslate(center_offset_pos,
                                            types='center_offset',
                                            image_size=self.image_size,
                                            grid_index=grid_index_to_x_y_axis)

                abs_double_pos = p_trans.abs_double_position.get_position()
                predict_kind_name = self.kinds_name[int(max_class_score_index)]
                res.append((predict_kind_name, abs_double_pos, prob_score))

        return res

    def predict(self, img) -> list:
        self.net.eval()
        if isinstance(img, np.ndarray):
            img = img.copy()
            if len(img.shape) != 3:
                print('img dimension must be 3! but get {}'.format(len(img.shape)))

            img = CV2.resize(img, new_size=self.image_size)
            img = CV2.cvtColorToRGB(img)

            x = np.transpose(img, axes=(2, 0, 1)) / 255.0
            x = torch.tensor([x], dtype=torch.float32).cuda()
        else:
            x = img.unsqueeze(0).cuda()

        y = self.net(x)
        y = y[0]
        y = y.cpu().detach().numpy().copy()
        return self.decode_out(y)
