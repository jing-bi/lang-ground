from PIL import Image, ImageDraw

def image_w_box(image,objxbox):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for label, boxes in objxbox.items():
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=2)
            label_text = f"{label}"
            bbox = draw.textbbox((x1, y1), label_text)
            draw.rectangle([bbox[0], bbox[1] - 5, bbox[2], bbox[3] + 5], fill="white")
            draw.text((x1, y1 - 5), label_text, fill="red")
    return image
