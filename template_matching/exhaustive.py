import numpy as np
import math
import cv2

P = 7
vidcap = cv2.VideoCapture('movie.mov')
success, image = vidcap.read()
c = 0
while success:
    cv2.imwrite("fr%d.jpg" % c, image)  # save fr as JPEG file
    success, image = vidcap.read()
    c += 1
ref = cv2.imread('reference.jpg', 0)

wei, hei = ref.shape[1], ref.shape[0]


def ex_h(img, ref, x, y):
    md = -1000000000
    ii_m = 0
    jj_m = 0
    for i in range((x - P), (x + P)):
        for j in range((y - P), (y + P)):
            diff = (np.sum((img[i:i + hei, j:j + wei] - ref) * (img[i:i + hei, j:j + wei] - ref)))
            if diff > md:
                md = diff
                ii_m = i
                jj_m = j
    return ii_m, jj_m


image_read = cv2.imread("fr0.jpg", 0)
md = -1000000000
x = 0
u = 0
for i in range(image_read.shape[0] - hei):
    for j in range(image_read.shape[1] - wei):
        diff = (np.sum((image_read[i:i + hei, j:j + wei] - ref) * (image_read[i:i + hei, j:j + wei] - ref)))
        if diff > md:
            md = diff
            x = i
            y = j

for i in range(1, c):
    image_read = cv2.imread("fr%d.jpg" % i, 0)
    x, y = ex_h(image_read, ref, x, y)
    cv2.rectangle(image_read, (y, x), (y + wei, x + hei), (255, 0, 0), 2)
    cv2.imwrite('out%d.jpg' % i, image_read)
