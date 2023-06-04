from PIL import Image, ImageOps
import PIL.ImageOps

for j in range(14, 21):  
    for i in range(20):
        image = Image.open("./trainingdata2/s{}/img{}.jpg".format(j, i+1))
        rotated_image = image.rotate(10)
        rotated_image.save("./trainingdata2/s{}/img{}_rotate1.jpg".format(j, i+1))

    for i in range(20):
        image = Image.open("./trainingdata2/s{}/img{}.jpg".format(j, i+1))
        rotated_image = image.rotate(20)
        rotated_image.save("./trainingdata2/s{}/img{}_rotate2.jpg".format(j, i+1))

    for i in range(20):
        image = Image.open("./trainingdata2/s{}/img{}.jpg".format(j, i+1))
        rotated_image = image.rotate(30)
        rotated_image.save("./trainingdata2/s{}/img{}_rotate3.jpg".format(j, i+1))

    for i in range(20):
        image = Image.open("./trainingdata2/s{}/img{}.jpg".format(j, i+1))
        rotated_image = image.rotate(-10)
        rotated_image.save("./trainingdata2/s{}/img{}_rotate4.jpg".format(j, i+1))

    for i in range(20):
        image = Image.open("./trainingdata2/s{}/img{}.jpg".format(j, i+1))
        rotated_image = image.rotate(-20)
        rotated_image.save("./trainingdata2/s{}/img{}_rotate5.jpg".format(j, i+1))

    for i in range(20):
        image = Image.open("./trainingdata2/s{}/img{}.jpg".format(j, i+1))
        rotated_image = image.rotate(-30)
        rotated_image.save("./trainingdata2/s{}/img{}_rotate6.jpg".format(j, i+1))

    for i in range(20):
        image = Image.open("./trainingdata2/s{}/img{}.jpg".format(j, i+1))
        inverse_image = ImageOps.mirror(image)
        inverse_image.save("./trainingdata2/s{}/img{}_inverse.jpg".format(j, i+1))




# from PIL import ImageEnhance, Image

# # 이미지를 밝게하는 함수
# def brighten_image(image, brightness):
#     enhancer = ImageEnhance.Brightness(image)
#     return enhancer.enhance(brightness)

# for i in range(10) :
#     for j in range(20) :
#         # 증강할 이미지 경로 설정
#         image_path = f"./trainingdata/s{i + 1}/img{j + 1}.jpg"

#         image = Image.open(image_path)
#         brightness_levels = [0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.15, 1.2, 1.3]

#         for brightness in brightness_levels:
#             brightened_image = brighten_image(image, brightness)
#             brightened_image.save(f"./trainingdata/s{i + 1}/img{j + 1}_brightened_{brightness}.jpg")



# ------------------- blur ---------------------------
# import cv2

# for i in range(10):
#     for j in range(20) :
#         # 이미지 읽어오기
#         src = cv2.imread('./trainingdata/s{}/img{}.jpg'.format(i+1, j+1))
#         for sigma in range(1, 4):
#             dst = cv2.GaussianBlur(src, (0, 0), sigma*2)

#             # 결과 이미지 저장
#             cv2.imwrite('./trainingdata/s{}/img{}_blur{}.jpg'.format(i+1, j+1, sigma), dst)

# import cv2
# import numpy as np

# for i in range(10):
#     for j in range(20) :
#     # 이미지 읽어오기
#         img = cv2.imread('./trainingdata/s{}/img{}.jpg'.format(i+1, j+1))

#         for k in range(1, 4):
#             mean = 0
#             var = 10000 * k
#             sigma = var ** 0.5
#             gaussian = np.random.normal(mean, sigma, img.shape)
#             gaussian = gaussian.reshape(img.shape)

#             # 노이즈 추가
#             noisy_image = img + gaussian

#             # 결과 이미지 저장
#             cv2.imwrite('./trainingdata/s{}/img{}_noise{}.jpg'.format(i+1, j+1, k), noisy_image)