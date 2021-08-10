import cv2
import os
import numpy as np

threshold = 0.01

selfie = cv2.imread("data/selfie.jpg", cv2.IMREAD_UNCHANGED)
kevin = cv2.imread("data/kevin.jpg", cv2.IMREAD_UNCHANGED)

result = cv2.matchTemplate(selfie, kevin, cv2.TM_CCOEFF_NORMED)

cv2.imwrite(os.path.join("data", "result.jpg"), result)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
w = kevin.shape[1]
h = kevin.shape[0]

d = cv2.rectangle(selfie, max_loc,
                  (max_loc[0] + w, max_loc[1] + h), (0, 255, 255), 2)


cv2.imwrite(os.path.join("data", "final.jpg"), selfie)

print("threhold: MIN -> " + str(min_val) + ", MAX -> " + str(max_val))

x, y = np.where(threshold >= result)

print(len(x), len(y))
