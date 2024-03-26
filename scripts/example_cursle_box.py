import numpy
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import supervision as sv
import numpy as np

from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator


sam = sam_model_registry["vit_b"](checkpoint="../sam_vit_b_01ec64.pth") #Light
# sam = sam_model_registry["vit_l"](checkpoint="../sam_vit_l_0b3195.pth")
# sam = sam_model_registry["vit_h"](checkpoint="../sam_vit_h_4b8939.pth") #heavy

# モデルをGPUに載せる
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
sam.to(device=DEVICE)

IMAGE_PATH = "../image_data/teisyoku.png"
image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
h,w,_ = image_rgb.shape

# マウスで範囲指定する
class BoundingBox:
    def __init__(self, image):
        self.x1 = -1
        self.x2 = -1
        self.y1 = -1
        self.y2 = -1
        self.image = image.copy()
        plt.figure()
        plt.connect("motion_notify_event", self.motion)
        plt.connect("button_press_event", self.press)
        plt.connect("button_release_event", self.release)
        self.ln_v = plt.axvline(0)
        self.ln_h = plt.axhline(0)
        plt.imshow(self.image)
        plt.show()

    # 選択中のカーソル表示
    def motion(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.ln_v.set_xdata(event.xdata)
            self.ln_h.set_ydata(event.ydata)
            self.x2 = event.xdata.astype("int16")
            self.y2 = event.ydata.astype("int16")
        if self.x1 != -1 and self.x2 != -1 and self.y1 != -1 and self.y2 != -1:
            plt.clf()
            plt.imshow(self.image)
            ax = plt.gca()
            rect = patches.Rectangle(
                (self.x1, self.y1),
                self.x2 - self.x1,
                self.y2 - self.y1,
                angle=0.0,
                fill=False,
                edgecolor="#00FFFF",
            )
            ax.add_patch(rect)
        plt.draw()

    # ドラッグ開始位置
    def press(self, event):
        self.x1 = event.xdata.astype("int16")
        self.y1 = event.ydata.astype("int16")

    # ドラッグ終了位置、表示終了
    def release(self, event):
        plt.clf()
        plt.close()

    def get_area(self):
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)
    

bounding_box = BoundingBox(image_rgb)
rt_x, rt_y, lb_x, lb_y = bounding_box.get_area()

box = np.array([rt_x, rt_y, lb_x, lb_y])
rect = patches.Rectangle((rt_x, rt_y), lb_x-rt_x, lb_y-rt_y, linewidth=1, edgecolor='r', facecolor='none')

mask_predictor = SamPredictor(sam)
mask_predictor.set_image(image_rgb)
masks, scores, logits = mask_predictor.predict(
    box=box,
    multimask_output=True
)

m = np.copy(image_rgb)
for _h in range(h):
    for _w in range(w):
        if masks[0][_h][_w] == True:
            m[_h][_w] = 0

print(masks[0])

# Add the patch to the Axes
plt.gca().add_patch(rect)

plt.imshow(m)
plt.title("bounding box Image")
plt.axis("off")  # 軸をオフにして、画像だけを表示します。
plt.show()