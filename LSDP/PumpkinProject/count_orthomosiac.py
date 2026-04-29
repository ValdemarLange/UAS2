import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window

filename = "orthomosaic_aksel.tif"
#filename = "pumpkinField.tif"

tile_size = 700 # Does not impact the results, only viewing pleasure
overlap = 100
step = tile_size - overlap

img_counter = 0
single_img_pumpkin_counter = 0
pumpkin_counter = 0

#13.1.2 - inRange Lab with k = 1.0
# lower_lab = np.array([160.71745333, 134.77633026, 157.47614285], dtype=np.uint8)
# upper_lab = np.array([212.63779999, 146.14132031, 179.54582696], dtype=np.uint8)

# inrange Lab with k = 1.5
lower_lab = np.array([147.73736666, 131.93508275, 151.95872182], dtype=np.uint8)
upper_lab = np.array([225.61788666, 148.98256783, 185.06324799], dtype=np.uint8)    

# inRange Lab with k = 2.0
# lower_lab = np.array([134.75728, 129.09383524, 146.4413008], dtype=np.uint8)
# upper_lab = np.array([238.59797333, 151.82381534, 190.58066901], dtype=np.uint8)

total_area = []

# 1=red, 2=blue, 3=green, 4=cyan, 5=yellow, 6=pink, 7=white, 8=black
color_list = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255), (0, 0, 0)]


# kernel
# kernel = np.ones((15, 15), np.uint8)
kernel = np.ones((2, 2), np.uint8)


stop_debugging = False

with rasterio.open(filename) as src:
    width = src.width
    height = src.height
    print(".tif width:", width, "height:", height)

    print("Number of bands:", src.count)
    print("Data type:", src.dtypes)

    for row_start in range(0, height, step):
        if stop_debugging:
            break

        for col_start in range(0, width, step):
            if stop_debugging:
                break

            row_end = min(row_start + tile_size, height)
            col_end = min(col_start + tile_size, width)

            tile_height = row_end - row_start
            tile_width = col_end - col_start

            window_location = Window.from_slices(
                (row_start, row_end),
                (col_start, col_end)
            )
    
            img = src.read(window=window_location)

             # Henriks code for convering to cv2
            # temp = img.transpose(1, 2, 0)
            # t2 = cv2.split(temp)
            # img_cv = cv2.merge([t2[2], t2[1], t2[0]])
            # img_lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)

            # Convert image to CIELab and also fix 4 bands instead of 3 bands as Henriks code handles
            img_rgb = img[:3, :, :]
            img_rgb = img_rgb.transpose(1, 2, 0)

            img_cv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            img_lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
            
            # Segment the image using the inRange function
            seg_lab = cv2.inRange(img_lab, lower_lab, upper_lab)

            if not np.any(seg_lab):
                continue

            # cv2.imshow("seg_lab", seg_lab)
            # cv2.waitKey(0)

            blurred = cv2.GaussianBlur(seg_lab, (5, 5), 0)
            _, blurred_binary = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY)


            # closed_image = cv2.morphologyEx(blurred_binary, cv2.MORPH_OPEN, kernel)
            closed_image = cv2.morphologyEx(blurred_binary, cv2.MORPH_CLOSE, kernel)


            # closed_image = cv2.morphologyEx(seg_lab, cv2.MORPH_CLOSE, kernel)


            # Locate contours.
            contours, hierarchy = cv2.findContours(
                closed_image,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # overlap images and count if origin of blob is inside the current image's half of the overlap
            left_margin = overlap / 2 if col_start > 0 else 0
            top_margin = overlap / 2 if row_start > 0 else 0
            right_margin = overlap / 2 if col_end < width else 0
            bottom_margin = overlap / 2 if row_end < height else 0



            # Draw a circle above the center of each of the detected contours.
            for contour in contours:
                M = cv2.moments(contour)
                if M['m00'] == 0:
                    continue

                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                area = cv2.contourArea(contour)

                total_area.append(area)

                average_pumpkin_area = 90  # adjust this experimentally

                estimated_pumpkins = max(1, round(area / average_pumpkin_area))

                if (
                    cx >= left_margin and
                    cx < tile_width - right_margin and
                    cy >= top_margin and
                    cy < tile_height - bottom_margin
                ):
                    # cv2.circle(img_cv, (cx, cy), 5, (0, 0, 255), 2)
                    cv2.circle(img_cv, (cx, cy), 5, color_list[min(estimated_pumpkins-1, len(color_list) - 1)], 2)

                    single_img_pumpkin_counter += estimated_pumpkins
                    # single_img_pumpkin_counter += 1


                # if cx > overlap/2 and cx < tile_height - overlap/2 and cy > overlap/2 and cy < tile_width - overlap/2:

                # cv2.circle(img_cv, (cx, cy), 5, (0, 0, 255), 2)

            pumpkin_counter += single_img_pumpkin_counter
            print("Number of detected pumpkin blobs in img #",img_counter,": %d" % single_img_pumpkin_counter)

            # ======== VIEWER ========
            # left = img_cv.copy()
            # right = cv2.cvtColor(closed_image, cv2.COLOR_GRAY2BGR)
        
            # if right.shape == (700, 700, 3):
            #     cv2.rectangle(right, (50, 50), (650, 650), (0, 0, 255), 2)

            # view = np.concatenate((left, right), axis=1)

            # cv2.namedWindow("Pumpkin tiles", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("Pumpkin tiles", 1200, 600)
            # cv2.imshow("Pumpkin tiles", view)
            
            # key = cv2.waitKey(0)

            # if key == 27:
            #     print("Escape key pressed")
            #     stop_debugging = True

            # ======== VIEWER ========

            img_counter += 1
            single_img_pumpkin_counter = 0

            # cv2.destroyAllWindows()

print("Number of detected pumpkin blobs: %d" % pumpkin_counter)
print("Average pumpkin area: %d" % np.mean(total_area))

