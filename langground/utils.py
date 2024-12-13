from PIL import Image
import supervision as sv
import numpy as np
from torch import tensor

def image_w_box(image,objxbox):

    box_annotator = sv.BoxCornerAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()

    xyxys = np.array([v.tolist() for boxes in objxbox.values() for v in boxes])
    labels = [l for l, boxes in objxbox.items() for v in boxes]
    unique_labels = list(objxbox.keys())
    class_id_map = {label: idx for idx, label in enumerate(unique_labels)}
    class_id = [class_id_map[label] for label in labels]

    masks = np.zeros((len(xyxys), image.shape[0], image.shape[1]), dtype=bool)
    for i, (x1, y1, x2, y2) in enumerate(xyxys):
        masks[i, int(y1):int(y2), int(x1):int(x2)] = labels[i]

    if len(xyxys) == 0:
        return image

    detections = sv.Detections(
        xyxy=xyxys,
        mask=masks,
        class_id=np.array(class_id),
    )

    annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)

    return annotated_image

if __name__ == '__main__':
    image = Image.open("assets/demo.jpeg")
    objxbox = {'computer monitor': [tensor([ 169.5367,  301.8970, 3045.2866, 2145.4736], device='cuda:0')], 'lamp': [tensor([3400.5979,  981.1383, 4102.7178, 2417.0103], device='cuda:0')], 'kettle': [tensor([4435.6953, 1981.3882, 5318.8530, 2972.8535], device='cuda:0')], 'table': [tensor([3108.2896, 2602.6494, 5795.3037, 4201.5000], device='cuda:0')], 'business card': [tensor([ 751.5681, 2817.4629,  945.1781, 2976.9883], device='cuda:0')], 'dog': [tensor([2155.5217, 2504.7114, 2562.2791, 3173.9731], device='cuda:0'), tensor([1013.7704, 2669.0864, 1560.3319, 3452.0579], device='cuda:0')], 'inkpad': [tensor([ 755.5402, 2983.9380,  962.8440, 3176.2158], device='cuda:0')], 'mouse': [tensor([2752.5286, 3038.9062, 3046.8740, 3297.1704], device='cuda:0')], 'tray': [tensor([3314.1667, 2722.6509, 4805.7476, 3684.2314], device='cuda:0')], 'computer keyboard': [tensor([ 203.7615, 2907.8442,  737.0474, 3416.8616], device='cuda:0')], 'laptop': [tensor([ 525.8097, 2439.1343, 2882.1917, 4261.9614], device='cuda:0')], 'keyboard': [tensor([ 659.9836, 3511.1763, 2828.9368, 4271.0059], device='cuda:0')], 'cookie': [tensor([4638.1128, 3625.8831, 5082.5796, 4013.4021], device='cuda:0')]}
    image_w_box(image, objxbox).show()
