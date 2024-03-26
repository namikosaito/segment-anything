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

IMAGE_PATH = "../image_data/dog.jpeg"
image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
h,w,_ = image_rgb.shape

mask_predictor = SamPredictor(sam)
mask_predictor.set_image(image_rgb)

rt_x = 70
rt_y = 247
lb_x = 629
lb_y = 926
box = np.array([rt_x, rt_y, 626, 926])
rect = patches.Rectangle((rt_x, rt_y), lb_x-rt_x, lb_y-rt_y, linewidth=1, edgecolor='r', facecolor='none')

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