
DATA_WIDTH = 416
DATA_HEIGHT = 416

CLASS_NUM  = 3

ANCHORS = {
    13: [[270, 254], [291, 179], [162, 304]],
    26: [[175, 222], [112, 235], [175, 140]],
    52: [[81, 118], [53, 142], [44, 28]]
}

ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS[13]],
    26: [x * y for x, y in ANCHORS[26]],
    52: [x * y for x, y in ANCHORS[52]],
}