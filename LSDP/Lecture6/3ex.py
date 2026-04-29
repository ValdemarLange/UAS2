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


matches = bruteforce.match(des_penguin, des_cover)
matches = sorted(matches, key=lambda x: x.distance)

out = cv2.drawMatches(img_penguin, kp_penguin, img_cover, kp_cover, matches, img_cover, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("output", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

