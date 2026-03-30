import cv2
import numpy as np
import matplotlib.pyplot as plt


img_left = cv2.imread('C:/ling/left.jpg')
img_right = cv2.imread('C:/ling/right.jpg')


img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
img_right_rgb = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)


gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)


print("Шукаємо локальні ознаки на фото")
sift = cv2.SIFT_create()


kp_left, des_left = sift.detectAndCompute(gray_left, None)
kp_right, des_right = sift.detectAndCompute(gray_right, None)

# --- 3. СПІВСТАВЛЕННЯ ОЗНАК ---
print("співставляємо знайдені точки")

bf = cv2.BFMatcher()
matches = bf.knnMatch(des_right, des_left, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"знайдено хороших збігів: {len(good_matches)}")


src_pts = np.float32([kp_right[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_left[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

print("рахуємо матрицю трансформації (RANSAC)")

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


print("генеруємо панорамне зображення")

height_left, width_left = img_left_rgb.shape[:2]


panorama_width = width_left + img_right_rgb.shape[1]
panorama_height = height_left


panorama = cv2.warpPerspective(img_right_rgb, H, (panorama_width, panorama_height))


panorama[0:height_left, 0:width_left] = img_left_rgb


gray_panorama = cv2.cvtColor(panorama, cv2.COLOR_RGB2GRAY)
_, thresh = cv2.threshold(gray_panorama, 1, 255, cv2.THRESH_BINARY)
coords = cv2.findNonZero(thresh)
x_max = np.max(coords[:,:,0])
cropped_panorama = panorama[:, :x_max]

plt.figure(figsize=(15, 8))
plt.imshow(cropped_panorama)
plt.title("результат зшивання")
plt.axis('off')
plt.show()