import numpy
import torch
import matplotlib.pyplot as plt
import cv2
import supervision as sv

from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator

IMAGE_PATH = "../image_data/dog.jpeg"

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# plt.imshow(image_rgb)
# plt.title("Original Image")
# plt.axis("off")  # 軸をオフにして、画像だけを表示します。
# plt.show()

sam = sam_model_registry["vit_b"](checkpoint="../sam_vit_b_01ec64.pth") #Light
# sam = sam_model_registry["vit_l"](checkpoint="../sam_vit_l_0b3195.pth")
# sam = sam_model_registry["vit_h"](checkpoint="../sam_vit_h_4b8939.pth") #heavy

# モデルをGPUに載せる
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
sam.to(device=DEVICE)

# パラメータの設定
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side = 30, # 画像の片側に沿ってサンプリングされる点の数
    pred_iou_thresh = 0.9, # モデルの予測したマスク品質を使用した、[0,1]のフィルタリングの閾値
    stability_score_thresh = 0.9, # モデルのマスク予測値を2値化するために使用されるカットオフの変化に対するマスクの安定性を使用 [0,1]のフィルタリング閾値
    crop_n_layers = 2, #実行するレイヤー数
    crop_n_points_downscale_factor = 1, # レイヤーnでサンプリングされたサイドごとのポイント数をcrop_n_points_downscale_factor**nでスケールダウン
    min_mask_region_area = 100 # 小さい面積のマスクの切断領域と穴を除去
)

result = mask_generator.generate(image_rgb)
mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)
detections = sv.Detections.from_sam(result)
annotated_image = mask_annotator.annotate(image_bgr, detections)

annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

plt.imshow(annotated_image_rgb)
plt.title("Annotated Image")
plt.axis("off")
plt.show()

print(result)