import math

import cv2
import numpy as np
from matplotlib import pyplot as plt

window_size =  5
vidcap = cv2.VideoCapture('movie.mov')


success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("D:/cv2out/frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  # print('Read a new frame: ', success)
  count += 1

reference = cv2.imread('reference.jpg', 0)
image_read = cv2.imread('D:/cv2out/frame500.jpg', 0)
w, h = reference.shape[::-1]





print(w, h)
print(reference.shape)

# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# for meth in methods:
def find_diff(theImage, reference, loop_i, loop_j):
  # cv2.imshow(theImage[280:340, 330:390,])
  max_diff = -math.inf
  max_i =0
  max_j =0
  for j in range(loop_j):
    for i in range(loop_i):
      im3 = theImage[i:i+h, j:j+w]
      diff = (np.sum((im3-reference)**2))
      if diff>max_diff:
        max_diff= diff
        max_i = i
        max_j = j
  return max_i, max_j


def find_diff_framed(theImage, reference, loop_i, loop_j, a_i, a_j):
  # cv2.imshow(theImage[280:340, 330:390,])
  # windowed_loop_i_s = a_i - window_size
  # windowed_loop_i_e = a_i + window_size
  # if windowed_loop_i_s>loop_i:
  #   windowed_loop_i_s=loop_i
  # if windowed_loop_i_s<0:
  #   windowed_loop_i_s=0
  # if windowed_loop_i_e > loop_i:
  #   windowed_loop_i_e = loop_i
  # if windowed_loop_i_e < 0:
  #   windowed_loop_i_e = 0

  windowed_loop_j_s = a_j - window_size
  windowed_loop_j_e = a_j + window_size
  if windowed_loop_j_s>loop_j:
    windowed_loop_j_s=loop_j
  if windowed_loop_j_s<0:
    windowed_loop_j_s=0
  if windowed_loop_j_e > loop_j:
    windowed_loop_j_e = loop_j
  if windowed_loop_j_e < 0:
    windowed_loop_j_e = 0


  max_diff = -math.inf

  max_i =0
  max_j =0
  for j in range(a_j-window_size, a_j+window_size+1):
    for i in range(a_i-window_size, a_i+window_size+1):
      im3 = theImage[i:i+h, j:j+w]
      diff = (np.sum((im3-reference)**2))
      if diff>max_diff:
        max_diff= diff
        max_i = i
        max_j = j
  return max_i, max_j




print(image_read.shape)
# im3 = image_read[0:h, 0:w]

loop_i = image_read.shape[0] - h
loop_j = image_read.shape[1] - w
print(reference.shape)

print(loop_i, loop_j)
# video = cv2.VideoWriter('video.avi',-1,1,(image_read.shape[0],image_read.shape[1]))

for i in range(count):
  image_read = cv2.imread('D:/cv2out/frame%d.jpg'%i, 0)
  if i == 0:
    a_i, a_j =find_diff(image_read, reference, loop_i, loop_j)
  else:
    a_i, a_j = find_diff_framed(image_read, reference, loop_i, loop_j, a_i, a_j)

  cv2.rectangle(image_read,(a_j, a_i), (a_j+w, a_i+h), (178,34,34), 2)
  cv2.imwrite('D:/cv2out/kaj_kore%d.jpg' % i, image_read)
  # video.write(image_read)
# cv2.destroyAllWindows()
# video.release()




















  #
  # image_read = image_read[a_i:a_i+h, a_j:a_j+w]
  # cv2.imshow('hah', image_read)
  # cv2.imwrite('D:/cv2out/kaj_kore.jpg', image_read)
  # cv2.imwrite('D:/cv2out/kaj_kore%d.jpg'%i, image_read)
# #
# print(a_i, a_j, "ai, aj")
# cv2.imwrite('D:/cv2out/kaj_kore.jpg', image_read)

# for i in range(759):
#   meth = cv2.TM_CCOEFF
#   image_read = cv2.imread('D:/cv2out/frame%d.jpg'%i, 0)
#   img = image_read.copy()
#   method = meth
#   # Apply reference Matching
#   res = cv2.matchreference(img,reference,method)
#   min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#   # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#   # if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#   #     top_left = min_loc
#   # else:
#   top_left = max_loc
#   bottom_right = (top_left[0] + w, top_left[1] + h)
#   cv2.rectangle(img,top_left, bottom_right, (178,34,34), 2)
#   # plt.subplot(121),plt.imshow(res,cmap = 'gray')
#   # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#   # plt.imshow(img)
#   # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#   # plt.suptitle(meth)
#   # plt.show()
#   cv2.imwrite('D:/cv2out/out%d.jpg' % i, img)
# # cv2.destroyAllWindows()
#
# # im1 = cv2.imread('D:/cv2out/out1.jpg', 0)
# # image_read = cv2.imread('D:/cv2out/out2.jpg', 0)
# # im3 = cv2.imread('D:/cv2out/out3.jpg', 0)
# # # height , width =  image_read.shape
# # # #


# for i in range(750):
#   img = cv2.imread('D:/cv2out/kaj_kore%d.jpg'%i, 0)
#   video.write(img)


