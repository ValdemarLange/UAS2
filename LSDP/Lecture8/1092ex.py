import numpy as np
import cv2
import icecream as ic

def decompose_essential_matrix(E):
    svd = np.linalg.svd(E)
    D = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    R1 = svd.U @ D @ svd.Vh # Vh is already transposed
    R2 = svd.U @ D.transpose() @ svd.Vh
    t1 = svd.U @ np.array([[0], [0], [1]])
    t2 = svd.U @ np.array([[0], [0], [-1]])

    return (R1, R2, t1, t2)


essential_matrix = np.array(
    [
        [-0.00300216, -0.43213014, -0.14442965],
        [0.33859683, 0.01502989, -0.60288981],
        [0.11929845, 0.54699833, -0.0246066],
    ]
)

(R1, R2, t1, t2) = decompose_essential_matrix(essential_matrix)

print("Rotation solution 1")
print(R1)
print("Rotation solution 2")
print(R2)

print("Translation solution 1")
print(t1)
print("Translation solution 2")
print(t2)

img1 = cv2.imread("08_estimating_camera_motion/input/my_photo-1.jpg")
img2 = cv2.imread("08_estimating_camera_motion/input/my_photo-2.jpg")
gray_image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
kp_image1, des_image1 = sift.detectAndCompute(gray_image1, None)
kp_image2, des_image2 = sift.detectAndCompute(gray_image2, None)

matches = bf.match(des_image1, des_image2)

points1_temp = []
points2_temp = []
for idx, m in enumerate(matches):
    points1_temp.append(kp_image1[m.queryIdx].pt)
    points2_temp.append(kp_image2[m.trainIdx].pt)
    

# Convert points1 and point2 to floats.
points1 = np.float32(points1_temp)
points2 = np.float32(points2_temp)


ransacReprojecThreshold = 1
confidence = 0.99
fundamental_matrix, fundamental_mask = cv2.findFundamentalMat(
    points1,
    points2,
    cv2.FM_RANSAC,
    confidence=confidence,
    ransacReprojThreshold=ransacReprojecThreshold,
)

print("Fundamental Matrix:\n", fundamental_matrix)

# print("Fundamental Mask:\n", fundamental_mask)

draw_matches = cv2.drawMatches(
            img1,
            kp_image1,
            img2,
            kp_image2,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

masked_matches = []
for i in range(len(matches)):
    if fundamental_mask[i, 0]:
        masked_matches.append(matches[i])

draw_masked_matches = cv2.drawMatches(
            img1,
            kp_image1,
            img2,
            kp_image2,
            masked_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

# cv2.imshow("matches", draw_matches)
# cv2.waitKey(0)

# cv2.imshow("masked matches", draw_masked_matches)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cameraMatrix = np.array([[704, 0, 637], [0, 704, 376], [0, 0, 1]])

essential_matrix, essential_mask = cv2.findEssentialMat(points1, points2, cameraMatrix, method=cv2.FM_RANSAC, prob=0.99, threshold=1)
print("Essential Matrix:\n", essential_matrix)