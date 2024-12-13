from langground import LangGround
import cv2

image = cv2.imread('./assets/demo.jpeg')
lg = LangGround()
lg.localize(image, "i'm thirsty")
