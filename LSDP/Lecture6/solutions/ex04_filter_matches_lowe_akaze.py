import cv2


def main():
    # Detect and show sift features
    img_penguin = cv2.imread("../input/sequence-penguin.jpg")
    img_cover = cv2.imread("../input/sequence-cover.jpg")
    assert img_penguin is not None, "Failed to load image."
    assert img_cover is not None, "Failed to load image."

    gray_penguin = cv2.cvtColor(img_penguin, cv2.COLOR_BGR2GRAY)
    gray_cover = cv2.cvtColor(img_cover, cv2.COLOR_BGR2GRAY)

    akaze = cv2.AKAZE_create()
    kp_penguin, des_penguin = akaze.detectAndCompute(gray_penguin, None)
    kp_cover, des_cover = akaze.detectAndCompute(gray_cover, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(des_penguin, des_cover, 2)

    # Filter matches using the Lowe's ratio test
    ratio_thresh = 0.7
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    img3 = cv2.drawMatches(
        img_penguin, kp_penguin, img_cover, kp_cover, good_matches, None, **draw_params
    )

    cv2.imwrite("../output/ex04_akaze_matching_features_lowe.png", img3)


main()
