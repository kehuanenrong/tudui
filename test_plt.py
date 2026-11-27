import matplotlib.pyplot as plt

img = plt.imread("data/logo.png")

print("Matplotlib对图片包装的类型为：", type(img))

plt.figure("logo")
plt.imshow(img)
plt.show()