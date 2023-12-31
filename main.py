# %%
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load
from time import sleep


# %% [markdown]
# # Prepare dataset

# %% [markdown]
# ### Machine Learning

# %%
dir = "dataset/"
image_train = []
label_train = []
image_test = []
label_test = []

for main_path in os.listdir(dir):
    paths = os.path.join(dir, main_path)
    for sub_paths in os.listdir(paths):
        sub_path = os.path.join(paths, sub_paths)
        for file_names in os.listdir(sub_path):
            file_name = os.path.join(sub_path, file_names)
            if file_names.lower().endswith((".JPG", ".PNG", ".jpg", ".png")):
                if paths == "dataset/Train":
                    image_train.append(file_name)
                    label_train.append(sub_paths)
                elif paths == "dataset/Test":
                    image_test.append(file_name)
                    label_test.append(sub_paths)


# %% [markdown]
# ### Preprocessing

# %%
# Preprocessing gambar
def preprocess_image(image_path, extract=None):
    image = cv2.imread(image_path)
    image_resize = cv2.resize(image, (100, 100))
    if extract == "canny":
        # canny ecxtractsi feature
        image_extract = cv2.Canny(image_resize, 50, 100) / 255.0
        return image_extract
    elif extract == "rgb":
        # spreating RGB
        B = image_resize[:,:,0]
        G = image_resize[:,:,1]
        R = image_resize[:,:,2]
        # meyamakan channel yang terpisah
        b_equi = cv2.equalizeHist(B)
        g_equi = cv2.equalizeHist(G)
        r_equi = cv2.equalizeHist(R)

        B_histo = cv2.calcHist([b_equi],[0], None, [256], [0,256]) 
        G_histo = cv2.calcHist([g_equi],[0], None, [256], [0,256])
        R_histo = cv2.calcHist([r_equi],[0], None, [256], [0,256])

        image_extract = cv2.merge([b_equi,g_equi,r_equi]) / 255.0
        return image_extract, B_histo, G_histo, R_histo
    

# %%
# Visualisasi image canny
def visualize_images(images, labels, title):
    plt.figure(figsize=(12, 12))
    plt.suptitle(title)

    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(labels[i])
        plt.axis("off")
    plt.show()

# Visualisasi image RGB histogram
def visualize_histograms(image, image_original):
    img = image_original[0]
    img = cv2.imread(img)
    img_resize = cv2.resize(img, (100, 100))
    image_extract, B_histo, G_histo, R_histo = image[0]
    #original channels
    
    bo_histo = cv2.calcHist([img_resize], [0], None, [256], [0, 256])
    go_histo = cv2.calcHist([img_resize], [1], None, [256], [0, 256])
    ro_histo = cv2.calcHist([img_resize], [2], None, [256], [0, 256])

    fig, axs = plt.subplots(4, 2, figsize=(15, 15))

    axs[0, 0].imshow(img)
    axs[0, 0].set_title("original image")

    axs[0, 1].imshow(image_extract)
    axs[0, 1].set_title("New image")

    axs[1, 0].plot(bo_histo, "b")
    axs[1, 0].set_title("blue original")

    axs[1, 1].plot(B_histo, "b")
    axs[1, 1].set_title("blue equalized")

    axs[2, 0].plot(go_histo, "g")
    axs[2, 0].set_title("green original")

    axs[2, 1].plot(G_histo, "g")
    axs[2, 1].set_title("green equlized")

    axs[3, 0].plot(ro_histo, "r")
    axs[3, 0].set_title("red original")

    axs[3, 1].plot(R_histo, "r")
    axs[3, 1].set_title("red equlized")

    plt.tight_layout()
    return plt.show()

    

# %% [markdown]
# ### Canny

# %%
preprocessed_images_train_canny = [preprocess_image(image_path, "canny") for image_path in image_train]
visualize_images(preprocessed_images_train_canny, label_train, "Training Images")

preprocessed_images_test_canny = [preprocess_image(image_path, "canny") for image_path in image_test]
visualize_images(preprocessed_images_test_canny, label_test, "Test Images")

# %%
label_encoder = LabelEncoder()
label_encoder.fit(["A", "B"])
y_train = label_encoder.transform(label_train)
y_test = label_encoder.transform(label_test)

# %%
x_train = [np.array(image).flatten() for image in preprocessed_images_train_canny]
x_test = [np.array(image).flatten() for image in preprocessed_images_test_canny]

# %%
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# %%
y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy_score:{accuracy * 100:.1f}%")
print(y_pred)

# %%
best_score = 0
best_n = 1

for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    if score > best_score:
        best_score = score
        best_y_pred = y_pred
        best_n = i

print(f"Best n_neighbors: {best_n}")
print(f"Best accuracy: {best_score * 100:.1f}%")
print(f"Best accuracy: {best_y_pred}")


# %%
dump(knn, "best_n.joblib")

# %%
# live testing
# Mengambil gambar
vid = cv2.VideoCapture(0)
while True:
    if not vid.isOpened():
        print("sorry, kamera kamu gagal di load")
        sleep(5)
    ret, frame = vid.read()
    cv2.imshow("frame", frame)
    # Pencet s jika ingin save gambar
    if cv2.waitKey(1) & 0xFF == ord("s"):
        file_name = "save_image.png"
        cv2.imwrite(filename=file_name, img=frame)
        print("image tersave")
        cv2.destroyAllWindows()
        break
    # Pencet q jika ingin keluar
    elif cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
vid.release()
cv2.destroyAllWindows()

img_original = cv2.imread(file_name)
img_process = preprocess_image(file_name, "canny")
img_process_flatten = np.array(img_process).flatten()
knn = load("best_n.joblib")
predict = knn.predict([img_process_flatten])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

ax1.imshow(img_original)
ax1.set_title("original image")

ax2.imshow(img_process)
if predict == 0:
    ax2.set_title("Model memprediksi A")
elif predict == 1:
    ax2.set_title("Model memprediksi B")
fig.suptitle("KNN", fontsize=24, fontweight="bold")
fig.show()

# %% [markdown]
# ### RGB histogram

# %%
preprocessed_images_train_rgb = [preprocess_image(image_path, "rgb") for image_path in image_train]
visualize_histograms(preprocessed_images_train_rgb, image_train)

preprocessed_images_test_rgb = [preprocess_image(image_path, "rgb") for image_path in image_test]
visualize_histograms(preprocessed_images_test_rgb, image_test)


