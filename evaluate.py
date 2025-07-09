import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json 
from sklearn.metrics import confusion_matrix,classification_report,auc,roc_curve

from utils.data_loader import preprocess_img ,get_image_path_and_labels

MODEL_PATH="models/breast_cancer_model.h5"
HISTORY_PATH="models/training_history.json"
DATA_ROOT="archive"

model=tf.keras.models.load_model(MODEL_PATH)
image_paths,labels=get_image_path_and_labels(DATA_ROOT)

import math,random
combined=list(zip(image_paths,labels))
random.shuffle(combined)
image_paths,labels=zip(*combined)

total=len(image_paths)
train_size=math.floor(0.7*total)
val_size=math.floor(0.15*total)

test_paths=image_paths[train_size+val_size:]
test_labels=labels[train_size+val_size:]

#test dataset
def create_dataset_from_list(image_paths,labels,batch_size=32,img_size=(150,150)):
    dataset=tf.data.Dataset.from_tensor_slices(image_paths,labels)
    dataset=dataset.map(lambda x,y:preprocess_image(x,y,img_size),nums_parallel_calls=tf.data.AUTOTUNE)
    dataset=dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


test_ds=create_dataset_from_list(image_paths,labels)

#predict
y_true=[]
y_scores=[]

for images,labels in test_ds:
    preds=model.predict(images)
    y_scores.extend(preds.ravel())
    y_true.extend(preds.numpy())

y_pred=[1 if p>0.5 else 0 for p in y_scores]


#confusion matrix
cm=confusion_matrix(y_true,y_pred)
plt.figure(figsize(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-IDC", "IDC"], yticklabels=["Non-IDC", "IDC"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion MAtrix")
plt.tight_layout()
plt.show()

#Classification report
print("\nClassification report:")
print(classification_report(y_true,y_pred,target_names=["NON-ID","IDC"]))

#roc curve
fpr,tpr,_=roc_curve(y_true,y_scores)
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

#training history
with open(HISTORY_PATH,"r") as f:
    history=json.load(f)

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.plot(history["accuracy"], label="Train Acc")
plt.plot(history["val_accuracy"], label="Val Acc")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

##Predictions
import random
sample_paths = random.sample(test_paths, 9)
sample_labels = [1 if '1' in p.split(os.sep)[-2] else 0 for p in sample_paths]

plt.figure(figsize=(10, 10))
for i, path in enumerate(sample_paths):
    img_raw = tf.io.read_file(path)
    img = tf.image.decode_png(img_raw, channels=3)
    img_resized = tf.image.resize(img, (150, 150)) / 255.0
    pred = model.predict(tf.expand_dims(img_resized, 0))[0][0]
    label = "IDC" if sample_labels[i] == 1 else "Non-IDC"
    pred_label = "IDC" if pred > 0.5 else "Non-IDC"
    color = "green" if label == pred_label else "red"

    plt.subplot(3, 3, i+1)
    plt.imshow(img.numpy().astype("uint8"))
    plt.title(f"True: {label} | Pred: {pred_label}", color=color)
    plt.axis("off")

plt.tight_layout()
plt.show()

