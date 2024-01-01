# KNN-with-extract-feature-canny
In this image classification, we will use the bisindo dataset which will classify A and B using KNN with canny feature extract.

## Canny Image
![Image](https://github.com/AcqmalFadhilla/KNN-with-extract-feature-canny/blob/master/output%20A.png)

![Image](https://github.com/AcqmalFadhilla/KNN-with-extract-feature-canny/blob/master/output%20B.png)

## Demo
Run the main file main.py in terminal
`python main.py`

To test,please run code below
```python
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
```

