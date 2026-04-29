import cv2


def main():
    # Detect and show akaze features
    img_penguin = cv2.imread("../input/sequence-penguin.jpg")
    img_cover = cv2.imread("../input/sequence-cover.jpg")
    assert img_penguin is not None, "Failed to load image."
    assert img_cover is not None, "Failed to load image."

    gray_penguin = cv2.cvtColor(img_penguin, cv2.COLOR_BGR2GRAY)
    gray_cover = cv2.cvtColor(img_cover, cv2.COLOR_BGR2GRAY)

    akaze = cv2.AKAZE_create()
    kp_penguin, des_penguin = akaze.detectAndCompute(gray_penguin, None)
    kp_cover, des_cover = akaze.detectAndCompute(gray_cover, None)

    bf = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)

    matches = bf.match(des_penguin, des_cover)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(
        img_penguin,
        kp_penguin,
        img_cover,
        kp_cover,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cv2.imwrite("../output/ex03_akaze_matching_features.png", img3)


main()
