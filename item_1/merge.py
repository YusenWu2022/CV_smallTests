
import numpy as np
import cv2
from cv2 import cv2

  
b = cv2.imread("./b.png",-1) 
g = cv2.imread("./g.png",-1)
r = cv2.imread("./r.png",-1)




merged = np.array([b, g, r])
merged = merged.transpose([1, 2, 0])


cv2.imwrite("./result.png",merged)
cv2.waitKey(0)
input(0)

