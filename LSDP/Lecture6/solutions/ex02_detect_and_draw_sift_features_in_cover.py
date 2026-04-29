import cv2


def main():
    # Detect and show sift features
    img = cv2.imread("../input/sequence-cover.jpg")
    assert img is not None, "Failed to load image."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp = sift.detect(gray)

    img = cv2.drawKeypoints(
        img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imwrite("../output/ex02_image_with_sift_features.png", img)


main()
