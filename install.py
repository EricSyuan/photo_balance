import numpy as np
from mtcnn import MTCNN
import cv2
import numpy

def ImgScaleUp(img, scale_percent=100):
    new_width = int(img.shape[1]*scale_percent/100)
    new_height = int(img.shape[0]*scale_percent/100)
    new_dim = (new_width, new_height)
    img = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
    return img


filename = "C:/Users/syuan/Pictures/2019-09/1568963498430.jpg"
image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
image = ImgScaleUp(image, scale_percent=50)
detector = MTCNN()
face_bound_box = sorted([i["box"] for i in detector.detect_faces(image)], key=lambda x: x[0])
face_dict = {}
face_img = {}
new_image = image.copy()


# def isRectagleOverlap(face_dict):
#     r1 = [face_dict[i][0], face_dict[i][1],
#           face_dict[i][0] + face_dict[i][2], face_dict[i][1] + face_dict[i][3]]
#     r2 = [face_dict[j][0], face_dict[j][1],
#           face_dict[j][0] + face_dict[j][2], face_dict[j][1] + face_dict[j][3]]
#     if ((r1[0] >= r2[2]) or (r1[2] <= r2[0]) \
#             or (r1[3] <= r2[1]) or (r1[1] >= r2[3])):
#         return False
#     else:
#         return True
#
#
# def OverlapArea(face_dict):
#     r1 = face_dict[i]  # x, y, w, h
#     r2 = face_dict[j]
#     if isRectagleOverlap(face_dict) == True:
#         intersect_width = ((r1[2] + r2[2]) - (r2[0] + r2[2] - r1[0]))
#         intersect_height = ((r1[3] + r2[3]) - (r1[1] + r1[3] - r2[1]))
#         intersect_area = intersect_height * intersect_width
#         r1_area = r1[2] * r1[3]
#         r2_area = r2[2] * r2[3]
#         overlap_rate_r1 = intersect_area / r1_area
#         overlap_rate_r2 = intersect_area / r2_area
#         return overlap_rate_r1, overlap_rate_r2

for num in range(len(face_bound_box)):    #框臉
    face_dict[num] = face_bound_box[num]  #建立dictionary
    face_img[num] = new_image[face_dict[num][1]:(face_dict[num][1]+face_dict[num][3]), face_dict[num][0]:(face_dict[num][0]+face_dict[num][2])]
    cv2.rectangle(new_image, face_bound_box[num], (0, 0, 255), 3)
    cv2.putText(new_image, f'ID:{num}', (face_bound_box[num][0], face_bound_box[num][1] + face_bound_box[num][3]),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)


#center & balance
photo_center = {} #整張照片的絕對中心點
center_dict = {}  #各框框中心點[x,y]
area_dict = {}    #框框面積 + 歸一化
distance = {}     #各框框中心點到絕對中心點的距離 + 歸一化

photo_center = {"width": image.shape[1]/2, "high":image.shape[0]/2}  #計算整張照片的中心點
print(photo_center)
for a in range(len(face_dict)):
    center_dict[a] = [face_dict[a][0] + face_dict[a][2]/2, face_dict[a][1] + face_dict[a][3]/2]
    area_dict[a] = (face_dict[a][2]/image.shape[1]) * (face_dict[a][3]/image.shape[0])
    distance[a]= numpy.sqrt(((((center_dict[a][0] - photo_center["width"])/image.shape[1])**2) + (((center_dict[a][1] - photo_center["high"])/image.shape[0])**2)))


balance_value = 0
for b in range(len(face_dict)):
    direction = center_dict[b][0] - photo_center["width"]
    balance_value +=  distance[b] * area_dict[b] * np.sign(direction)
print(balance_value)

if abs(balance_value) <=0.003:
    print("it's balance")
elif abs(balance_value) >0.003 and balance_value <0:
    print("it's not balance,left of center")
elif abs(balance_value) >0.003 and balance_value >0:
    print("it's not balance,right of center")






# for i in range(len(face_dict)):
#     for j in range(i + 1, len(face_dict)):
#         if isRectagleOverlap(face_dict) == True:
#             print(f'ID {i} & {j} overlapped:', isRectagleOverlap(face_dict))   #顯示是否重疊
#             print(OverlapArea(face_dict))
#         else:
#             continue


print('face dict', face_dict)
cv2.imshow('mtcnn', cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)


