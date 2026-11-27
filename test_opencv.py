import cv2
import requests

# 示例图片下载链接(无需关注）
url = "https://pytorch.org/tutorials/_static/img/thumbnails/cropped/profiler.png"

# 下载图片（无需关注）
r = requests.get(url)
with open("logo.png", "wb") as f:
    f.write(r.content)

# 读取图片
img = cv2.imread("logo.png")

# 打印图片
print("OpenCV对图片包装的类型为：", type(img))

# 显示图片
cv2.imshow("logo", img)

# 直到按下任意键，图片才退出.  
cv2.waitKey(0)