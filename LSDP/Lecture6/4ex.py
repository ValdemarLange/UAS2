import cv2

img_penguin = cv2.imread("input/sequence-penguin.jpg")
if img_penguin is None:
    raise ValueError(f"Failed to load image")
img_cover = cv2.imread("input/sequence-cover.jpg")
if img_cover is None:
    raise ValueError(f"Failed to load image")


gray_penguin = cv2.cvtColor(img_penguin, cv2.COLOR_BGR2GRAY)
gray_cover = cv2.cvtColor(img_cover, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp_penguin, des_penguin = sift.detectAndCompute(gray_penguin, None)
kp_cover, des_cover = sift.detectAndCompute(gray_cover, None)

bruteforce = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
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

img3 = cv2.resize(img3, (0, 0), fx=0.7, fy=0.7)

cv2.imshow("output", img3)

cv2.waitKey(0)
cv2.destroyAllWindows()

