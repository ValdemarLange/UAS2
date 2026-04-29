import numpy as np
import cv2
import matplotlib.pyplot as plt
from lsdp_tools import FrameIterator

p1 = []


def main():
    fi = FrameIterator('input/remember.webm')
    generator = fi.frame_generator()

    foreground = np.array([[177, 73], [256, 146]])
    background = np.array([[456, 378], [581, 306]])
    
    # store pixel values over time
    fg_values = [[] for _ in range(len(foreground))]
    bg_values = [[] for _ in range(len(background))]

    acumulator = None
    mean_image = None
    counter = 0

    for frame in generator:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if acumulator is None:
            acumulator = frame.astype(float)
        counter += 1
        for i, point in enumerate(foreground):
            x, y = point
            cv2.circle(frame, (x, y), 4, (255, 0, 0), 2)
            fg_values[i].append(frame[y, x].copy())   # BGR pixel

        for i, point in enumerate(background):
            x, y = point
            cv2.circle(frame, (x, y), 3, (0, 0, 255), 2)
            bg_values[i].append(frame[y, x].copy())   # BGR pixel

        acumulator += frame
        mean_image = acumulator / counter

        running_mean = frame.astype(float)
        cv2.accumulateWeighted(frame, running_mean, 0.1)

        diff_image = frame - mean_image

        running_diff = cv2.absdiff(frame, cv2.convertScaleAbs(running_mean))
        

        # cv2.imshow('running_mean', running_mean)

        # cv2.imshow('diff_image', running_diff)
        # cv2.imshow('diff_image', diff_image)
        thresholded, changes = cv2.threshold(np.abs(diff_image), 50, 255, cv2.THRESH_BINARY)
        cv2.imshow('motion', changes)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    all_series = [
        ('Foreground', foreground, fg_values),
        ('Background', background, bg_values),
    ]

    plot_idx = 0
    for label, points, series_list in all_series:
        for i, values in enumerate(series_list):
            p = np.array(values)
            ax = axes[plot_idx]
            ax.plot(p[:, 2], label='R')
            ax.plot(p[:, 1], label='G')
            ax.plot(p[:, 0], label='B')
            ax.set_title(f'{label} point {i}: {points[i]}')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Pixel value')
            ax.legend()
            plot_idx += 1

    plt.tight_layout()
    plt.show()

main()