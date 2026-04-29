import cv2
import numpy as np


def main():
    # Detect and show sift features
    img_penguin = cv2.imread("../input/sequence-penguin.jpg")
    img_cover = cv2.imread("../input/sequence-cover.jpg")
    assert img_penguin is not None, "Failed to load image."
    assert img_cover is not None, "Failed to load image."

    gray_penguin = cv2.cvtColor(img_penguin, cv2.COLOR_BGR2GRAY)
    gray_cover = cv2.cvtColor(img_cover, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp_penguin, des_penguin = sift.detectAndCompute(gray_penguin, None)
    kp_cover, des_cover = sift.detectAndCompute(gray_cover, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(des_penguin, des_cover)

    src_pts = np.float32([kp_penguin[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_cover[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 50.0)

    matchesMask = mask.ravel().tolist()

    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask,  # draw only inliers
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    img3 = cv2.drawMatches(
        img_penguin, kp_penguin, img_cover, kp_cover, matches, None, **draw_params
    )

    cv2.imwrite("../output/ex05_sift_matching_features_homography.png", img3)


main()
