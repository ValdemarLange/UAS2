import cv2
import numpy as np


def main():
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    agb = cv2.aruco_GridBoard()
    print(agb)
    test = agb.create(6, 10, 60, 20, dictionary=aruco_dict)
    print(test)
    print(test.getMarkerSeparation())
    print(test.getMarkerLength())
    print(test.getGridSize())
    img = test.draw((1000, 2000))
    img2 = cv2.aruco.drawPlanarBoard(test, (2000, 3000), marginSize=50,
            borderBits = 1)
    print(test)
    filename = "aruco_grid_board_dict_4x4_250.png"
    cv2.imwrite(filename, img2)



main()
