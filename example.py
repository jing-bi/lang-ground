from langground import LangGround
import cv2
import time

image = cv2.imread('./assets/demo.jpeg')
lg = LangGround()
# lg.localize(image, "i'm thirsty")
while True:
    start = time.time()
    res = lg.select("i am thirsty", ["bottle", "water", "cup", "laptop"], block=True)
    print(time.time() - start)
    time.sleep(0.5)
    print(res)
