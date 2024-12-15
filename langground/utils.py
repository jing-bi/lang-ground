from PIL import Image
import supervision as sv
import numpy as np
from torch import tensor
import cv2

colors = sv.ColorPalette.from_hex(
    [
        "#a1c9f4",
        "#ffb482",
        "#8de5a1",
        "#ff9f9b",
        "#d0bbff",
        "#debb9b",
        "#fab0e4",
        "#cfcfcf",
        "#fffea3",
        "#b9f2f0",
        "#a1c9f4",
        "#ffb482",
        "#8de5a1",
        "#ff9f9b",
        "#d0bbff",
        "#debb9b",
        "#fab0e4",
        "#cfcfcf",
        "#fffea3",
        "#b9f2f0",
    ]
)

text_palette = {str(idx): colors.by_idx(idx).as_hex() for idx in range(50)}


def image_w_box(image,objxbox):

    box_annotator = sv.BoxCornerAnnotator(thickness=10, corner_length=30, color=colors)
    label_annotator = sv.LabelAnnotator(color=colors)
    mask_annotator = sv.MaskAnnotator(opacity=0.2, color=colors)

    xyxys = np.array([v.tolist() for boxes in objxbox.values() for v in boxes])
    unique_labels = sorted(objxbox.keys())
    class_id_map = dict(enumerate(unique_labels))
    labels = [l for l, boxes in objxbox.items() for _ in boxes]
    class_id = [list(class_id_map.values()).index(label) for label in labels]

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
    # Convert RGB to BGR for annotation
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # After annotation, convert back to RGB
    annotated_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)

    return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)


def visualize_masks(frame: np.ndarray, masks, obj_ids, obj_to_question) -> np.ndarray:
    """Visualize masks on the frame with different colors for each object."""
    result = frame.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

    for mask, obj_id in zip(masks, obj_ids):
        q_idx = obj_to_question[obj_id]
        color = colors[q_idx % len(colors)]
        mask_colored = np.zeros_like(frame)
        mask_colored[mask.squeeze()] = color
        result = cv2.addWeighted(result, 1, mask_colored, 0.5, 0)

    return result


def anno_frame(frame, segments, query, obj_to_question):

    if segments:
        masks = list(segments.values())
        labels = list(segments.keys())
        frame = visualize_masks(frame, masks, labels, obj_to_question)

    return frame


def image_w_box_cv2(image, objxbox):
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a NumPy array.")

    image_copy = image.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX

    height, width, _ = image.shape
    font_scale = max(0.5, min(width, height) / 1000)  
    font_thickness = max(1, int(font_scale * 2))  

    for label, boxes in objxbox.items():
        for box in boxes:
            print("box", box)

            x1, y1, x2, y2 = map(int, box.tolist())

            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

            label_text = f"{label}"

            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, font, font_scale, font_thickness
            )

            text_x1 = x1
            text_y1 = y1 - text_height - baseline
            text_x2 = x1 + text_width
            text_y2 = y1

            cv2.rectangle(image_copy, (text_x1, text_y1), (text_x2, text_y2), color=(255, 255, 255), thickness=-1)

            cv2.putText(
                image_copy,
                label_text,
                (x1, y1 - baseline),
                font,
                font_scale,
                color=(0, 0, 255),
                thickness=font_thickness,
                lineType=cv2.LINE_AA,
            )

    return image_copy

if __name__ == '__main__':
    image = Image.open("assets/demo.jpeg")
    objxbox = {'computer monitor': [tensor([ 169.5367,  301.8970, 3045.2866, 2145.4736], device='cuda:0')], 'lamp': [tensor([3400.5979,  981.1383, 4102.7178, 2417.0103], device='cuda:0')], 'kettle': [tensor([4435.6953, 1981.3882, 5318.8530, 2972.8535], device='cuda:0')], 'table': [tensor([3108.2896, 2602.6494, 5795.3037, 4201.5000], device='cuda:0')], 'business card': [tensor([ 751.5681, 2817.4629,  945.1781, 2976.9883], device='cuda:0')], 'dog': [tensor([2155.5217, 2504.7114, 2562.2791, 3173.9731], device='cuda:0'), tensor([1013.7704, 2669.0864, 1560.3319, 3452.0579], device='cuda:0')], 'inkpad': [tensor([ 755.5402, 2983.9380,  962.8440, 3176.2158], device='cuda:0')], 'mouse': [tensor([2752.5286, 3038.9062, 3046.8740, 3297.1704], device='cuda:0')], 'tray': [tensor([3314.1667, 2722.6509, 4805.7476, 3684.2314], device='cuda:0')], 'computer keyboard': [tensor([ 203.7615, 2907.8442,  737.0474, 3416.8616], device='cuda:0')], 'laptop': [tensor([ 525.8097, 2439.1343, 2882.1917, 4261.9614], device='cuda:0')], 'keyboard': [tensor([ 659.9836, 3511.1763, 2828.9368, 4271.0059], device='cuda:0')], 'cookie': [tensor([4638.1128, 3625.8831, 5082.5796, 4013.4021], device='cuda:0')]}
    image_w_box(image, objxbox).show()
