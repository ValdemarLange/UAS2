import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../input/DJI_0222.JPG")
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)

# visualize channels
plt.figure()
plt.title("L channel"); plt.imshow(L, cmap="gray"); plt.axis("off")
plt.figure()
plt.title("a channel"); plt.imshow(A, cmap="gray"); plt.axis("off")
plt.figure()
plt.title("b channel"); plt.imshow(B, cmap="gray"); plt.axis("off")

# Try some candidate masks (adjust thresholds)
mask1 = (A > 150).astype(np.uint8) * 255          # single channel
mask2 = (B > 160).astype(np.uint8) * 255          # single channel
mask3 = ((A.astype(np.int16) - B.astype(np.int16)) > 10).astype(np.uint8) * 255  # linear combo
mask4 = ((B.astype(np.int16) - A.astype(np.int16)) > 10).astype(np.uint8) * 255

plt.figure(); plt.title("mask1 (A>150)"); plt.imshow(mask1, cmap="gray"); plt.axis("off")
plt.figure(); plt.title("mask2 (B>160)"); plt.imshow(mask2, cmap="gray"); plt.axis("off")
plt.figure(); plt.title("mask3 (A-B>10)"); plt.imshow(mask3, cmap="gray"); plt.axis("off")
plt.figure(); plt.title("mask4 (B-A>10)"); plt.imshow(mask4, cmap="gray"); plt.axis("off")
plt.show()
