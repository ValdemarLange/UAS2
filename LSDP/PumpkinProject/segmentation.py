import cv2
import numpy as np
import matplotlib.pyplot as plt

file = "image.png" # Original image is nr 0110 in the dataset
file_annot = "annotation.png" # Annotated in gimp

# Annotated with blue (255,0,0 in bgr)
ANNOT_LOWER_BGR = (245, 0, 0)
ANNOT_UPPER_BGR = (255, 10, 10)

img_bgr = cv2.imread(file)
annot_bgr = cv2.imread(file_annot)

# ============================================================
# 13.1.1 - EXTRACT ANNOTATED PUMPKIN PIXELS

# Create mask from blue annotation
annot_mask = cv2.inRange(
    annot_bgr,
    ANNOT_LOWER_BGR,
    ANNOT_UPPER_BGR
)

print("Annotated pixels:", np.sum(annot_mask == 255))

# ============================================================
# 13.1.1 - BGR COLOUR STATISTICS

mean_bgr, std_bgr = cv2.meanStdDev(img_bgr, mask=annot_mask)
std_bgr = std_bgr.flatten()

print("\n13.1.1 - BGR statistics")
print("Mean BGR:", mean_bgr.flatten())
print("Std BGR:", std_bgr.flatten())

# Extract annotated pixels manually too
pixels_bgr = img_bgr.reshape((-1, 3))
mask_pixels = annot_mask.reshape((-1))

annot_pixels_bgr = pixels_bgr[mask_pixels == 255]

# ============================================================
# 13.1.1 - CIELAB COLOUR STATISTICS

img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

mean_lab, std_lab = cv2.meanStdDev(img_lab, mask=annot_mask)

L_standard = mean_lab[0] * 100 / 255
a_standard = mean_lab[1] - 128
b_standard = mean_lab[2] - 128

print("\n13.1.1 - CieLAB statistics")
print("Mean LAB:", L_standard, a_standard, b_standard)
print("Std Lab:", std_lab.flatten(), "Probably in wrong format/scale")

pixels_lab = img_lab.reshape((-1, 3))
annot_pixels_lab = pixels_lab[mask_pixels == 255]

avg_lab = np.average(annot_pixels_lab, axis=0)
cov_lab = np.cov(annot_pixels_lab.T)

print("\nAverage pumpkin colour Lab:", avg_lab)


# 13.1.1 - VISUALISE COLOUR DISTRIBUTION ### I HAVE NO IDEA IF THIS IS WHAT HENRIK WANTED

# BGR distribution in red/green.
plt.figure()
plt.plot(annot_pixels_bgr[:, 1], annot_pixels_bgr[:, 2], ".")
plt.title("Annotated pumpkin pixels in RGB space")
plt.xlabel("Green [0-255]")
plt.ylabel("Red [0-255]")
plt.tight_layout()
plt.savefig("13_1_1_rgb_distribution.pdf", dpi=150)
# plt.show()


plt.figure()
plt.plot(annot_pixels_lab[:, 1], annot_pixels_lab[:, 2], ".")
plt.title("Annotated pumpkin pixels in CieLAB space")
plt.xlabel("a channel")
plt.ylabel("b channel")
plt.tight_layout()
plt.savefig("13_1_1_lab_distribution.pdf", dpi=150)
# plt.show()


# ============================================================
# 13.1.2 - inRange WITH RGB/BGR VALUES

# k = how many standard deviations from the mean
k = 1.0

lower_bgr = mean_bgr.flatten() - k * std_bgr
upper_bgr = mean_bgr.flatten() + k * std_bgr

print("\n13.1.2 - inRange BGR")
print("Lower BGR:", lower_bgr)
print("Upper BGR:", upper_bgr)

seg_bgr = cv2.inRange(img_bgr, lower_bgr, upper_bgr)

cv2.imwrite("13_1_2_segmentation_inrange_bgr.png", seg_bgr)


# ============================================================
# 13.1.2 - inRange WITH CIELAB VALUES

lower_lab = avg_lab - k * std_lab.flatten()
upper_lab = avg_lab + k * std_lab.flatten()


print("\n13.1.2 - inRange Lab")
print("Lower Lab:", lower_lab)
print("Upper Lab:", upper_lab)

seg_lab = cv2.inRange(img_lab, lower_lab, upper_lab)

cv2.imwrite("13_1_2_segmentation_inrange_lab.png", seg_lab)

# plt.figure()
# plt.imshow(seg_lab, cmap="gray")
# plt.title("13.1.2 - inRange using CieLAB values")
# plt.axis("off")
# plt.show()



# ============================================================
# 13.1.2 - DISTANCE IN RGB SPACE


# Compare every pixel to the average pumpkin colour.
diff_bgr = pixels_bgr.astype(float) - mean_bgr.flatten()

# Squared Euclidean distance
dist_bgr = np.sum(diff_bgr ** 2, axis=1)

dist_bgr_image = dist_bgr.reshape((img_bgr.shape[0], img_bgr.shape[1]))

threshold_dist = 2500

seg_dist = dist_bgr_image < threshold_dist

output_dist = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
output_dist[seg_dist] = 255

cv2.imwrite("13_1_2_segmentation_distance_bgr.png", output_dist)

# plt.figure()
# plt.imshow(output_dist, cmap="gray")
# plt.title("13.1.2 - Distance in RGB/BGR space")
# plt.axis("off")
# plt.show()

# Maybe use the mahalanobis distance instead?


