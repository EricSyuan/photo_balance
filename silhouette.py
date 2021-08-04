import numpy as np
from mtcnn import MTCNN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import cv2


def ImgScaleUp(img, scale_percent=100):   #放大縮小照片
    new_width = int(img.shape[1]*scale_percent/100)
    new_height = int(img.shape[0]*scale_percent/100)
    new_dim = (new_width, new_height)
    img = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
    return img


filename = "D:/picture/2019-09/line_663491189131412.jpg"
image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) #讀取照片
image = ImgScaleUp(image, scale_percent=100)   #設定照片大小
detector = MTCNN()
face_bound_box = sorted([i["box"] for i in detector.detect_faces(image)], key=lambda x: x[0])
face_dict = {}
face_img = {}
new_image = image.copy()




for num in range(len(face_bound_box)):    #框臉
    face_dict[num] = face_bound_box[num]  #建立dictionary
    face_img[num] = new_image[face_dict[num][1]:(face_dict[num][1]+face_dict[num][3]), face_dict[num][0]:(face_dict[num][0]+face_dict[num][2])]
    cv2.rectangle(new_image, face_bound_box[num], (0, 0, 255), 3)
    # cv2.putText(new_image, f'ID:{num}', (face_bound_box[num][0], face_bound_box[num][1] + face_bound_box[num][3]),
    #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)


#center & balance
photo_center = {} #整張照片的絕對中心點
center_dict = {}  #各框框中心點[x,y]
area_dict = {}    #框框面積 + 歸一化
distance = {}     #各框框中心點到絕對中心點的距離 + 歸一化


photo_center = {"width": image.shape[1]/2, "high":image.shape[0]/2}  #計算整張照片的中心點

print(photo_center['width'])
print(photo_center['high'])
for a in range(len(face_dict)):
    center_dict[a] = [face_dict[a][0] + face_dict[a][2]/2, face_dict[a][1] + face_dict[a][3]/2]


center_list = []
for b in range(len(face_dict)):
    center_list.append(center_dict[b])
print('center list:',center_list)

scores = []
df=np.array(center_list)
print('df:',df)
K_max= len(face_dict)
for c in range (2 ,K_max):
    k = KMeans(n_clusters=c).fit(df)
    #distortions.append(kmeans.inertia_)  # 誤差平方和 (SSE)
    scores.append(silhouette_score(df, KMeans(n_clusters=c).fit_predict(df)))  # 側影係數

#判定當人數低於2人時的分組處理
if len(scores)==0:
    scores.append(-1)

# 找出最大的側影係數來決定 K 值
if scores.index(max(scores)) == 0:   #壓低k=2 的誤差
    scores[np.argmax(scores)]= np.min(scores)
print(scores)
selected_K = scores.index(max(scores)) + 2  #決定k值
print('K =', selected_K)

# 建立 KMeans 模型並預測目標值
kmeans = KMeans(n_clusters=selected_K).fit(df)
new_dy = kmeans.predict(df)
print('new_dy',new_dy)
# 資料分組的中心點   !!!注意不是Global的中心
new_centers = kmeans.cluster_centers_
print('new_centers',new_centers)
# 新資料分組繪圖
plt.subplot(222)
plt.title(f'KMeans={selected_K} groups')
plt.scatter(df.T[0], df.T[1], c=new_dy, cmap=plt.cm.Set3)
plt.scatter(new_centers.T[0], new_centers.T[1], marker='^', color='orange')
for i in range(new_centers.shape[0]): # 標上各分組中心點
    plt.text(new_centers.T[0][i], new_centers.T[1][i], str(i + 1),
             fontdict={'color': 'red', 'weight': 'bold', 'size': 24})




#繪製側影(輪廓)係數圖: silhouette_score
k_range =range(2 ,K_max)
#plt.subplot(224)
plt.title('Silhouette score')
plt.subplot(224)
plt.plot(k_range , scores)
plt.plot(selected_K, scores[selected_K - 2], 'go') # 最佳解
plt.tight_layout()
plt.show()


wcss = []   #手肘法圖示
for i in range(1, len(face_dict)+1):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(center_list)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, len(face_dict)+1), wcss)
plt.show()

#繪製圖片
cv2.imshow('mtcnn', cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
