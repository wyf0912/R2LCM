import numpy as np
import cv2
import glob
imgs = [
    "error_map/WACV_fivek_dataset_val_jpeg_-1_2.0e-04/31.jpg",
        "error_map/WACV_fivek_dataset_val_jpeg_-1_2.0e-04/33.jpg",
            "error_map/WACV_fivek_dataset_val_jpeg_-1_2.0e-04/51.jpg",
]
for idx, img_path in enumerate(imgs):
    np_img = cv2.imread(img_path)
    np_img_gray = np_img.mean(axis=2)
    w_sum = np.sum(np_img_gray, axis=0)
    h_sum = np.sum(np_img_gray, axis=1)
    h_area = np.argwhere(h_sum < h_sum.max()-2500)
    w_area = np.argwhere(w_sum < w_sum.max()-2500)
    cv2.imwrite(f"cropped_{idx}.jpg", np_img[h_area.min():h_area.max(), w_area.min():w_area.max()])
    pass