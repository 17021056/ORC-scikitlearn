import cv2
from PIL import Image
import numpy as np
import os
import glob
from skimage.feature import hog
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Load dữ liệu training
data_path = os.path.join("X_train",'*g')
files = glob.glob(data_path)
X_train = []
for f1 in files:
    img = Image.open(f1)
    arr = np.array(img)
    X_train.append(arr)
#load nhãn cho dữ liệu training
y_train = np.loadtxt("labels.txt")
# load dữ liệu test
data_pathtest = os.path.join("X_test",'*g')
filestest = glob.glob(data_pathtest)
X_test = []
for f2 in filestest:
    img = Image.open(f2)
    arrtest = np.array(img)
    X_test.append(arrtest)
#load nhãn cho dữ liệu test
y_test = np.loadtxt("labelstest.txt")
#cho x_train
X_train_feature = []
for i in range(len(X_train)):
    feature = hog(X_train[i],orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),block_norm="L2")
    X_train_feature.append(feature)
X_train_feature = np.array(X_train_feature,dtype = np.float32)
#cho x_test
X_test_feature = []
for i in range(len(X_test)):
    feature = hog(X_test[i],orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),block_norm="L2")
    X_test_feature.append(feature)
X_test_feature = np.array(X_test_feature,dtype=np.float32)
#tính độ sai số C cho xác suất đúng cao nhất
parameter_candidates = [
  {'C': [0.001, 0.01, 0.1, 1, 5, 10, 100, 1000], 'kernel': ['linear']},
]
model = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, n_jobs=-1)
# model = LinearSVC(C=5)
model.fit(X_train_feature,y_train)
print('Best score:', model.best_score_)
print('Best C:',model.best_estimator_.C)


#đoán bộ test
y_pre = model.predict(X_test_feature)
# print(y_pre)
print(accuracy_score(y_test,y_pre))
# cho ảnh cần detect vào để đoán
image = cv2.imread("digit.jpg")
im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
im_blur = cv2.GaussianBlur(im_gray,(5,5),0)
im,thre = cv2.threshold(im_blur,90,255,cv2.THRESH_BINARY_INV)
_,contours,hierachy = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(cnt) for cnt in contours]
for i in contours:
        (x, y, w, h) = cv2.boundingRect(i)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        roi = thre[y:y + h, x:x + w]
        roi = np.pad(roi, (20, 20), 'constant', constant_values=(0, 0))
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        # dilate thêm pixel vào biên của đối tượng
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), block_norm="L2")

        nbr = model.predict(np.array([roi_hog_fd], np.float32))
        print(nbr)
        # đặt text vào ảnh
        cv2.putText(image, str(int(nbr[0])), (x+int(w/2), y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("image", image)
        print("detect complete !!")
cv2.imwrite("image_pand.png", image)
cv2.waitKey()
cv2.destroyAllWindows()
