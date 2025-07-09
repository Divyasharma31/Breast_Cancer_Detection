from utils.data_loader import create_dataset, get_image_path_and_labels,preprocess_img
from utils.model_builder import build_simple_cnn
import tensorflow as tf
DATA_ROOT="archive"
#splitting manuallly
import math

image_paths,labels=get_image_path_and_labels(DATA_ROOT)
total_samples=len(image_paths)

train_size=math.floor(0.7*total_samples)
val_size=math.floor(0.15*total_samples)
test_size=total_samples - train_size - val_size

#time to shuffle
import random
combined=list(zip(image_paths,labels))
random.shuffle(combined)
image_paths[:],labels[:]=zip(*combined)

# creating train variables 
train_paths=image_paths[:train_size]
train_labels=labels[:train_size]
# creating val var
val_paths=image_paths[train_size:train_size+val_size]
val_labels=labels[train_size:train_size+val_size]

# creating test var 
test_paths=image_paths[train_size+val_size:]
test_labels=labels[train_size+val_size:]
#as it was taking a lot of time every time to process all we will use a small portion to showcase
image_paths=image_paths[:5000]
labels=labels[:5000]


def create_dataset_from_list(image_paths,labels,batch_size=32,shuffle=True,buffer_size=10000,img_size=(150,150)):
    dataset=tf.data.Dataset.from_tensor_slices((image_paths,labels))
    dataset=dataset.map(lambda x,y: preprocess_img(x,y,img_size),num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset=dataset.shuffle(buffer_size)
    dataset=dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# creating dataset variables from the ones we initialized earlier
train_ds=create_dataset_from_list(train_paths,train_labels)
val_ds=create_dataset_from_list(val_paths,val_labels)
test_ds=create_dataset_from_list(test_paths,test_labels)


#checking the model
model=build_simple_cnn()
# epochs=10, it will go through model 10 times
history=model.fit(train_ds,validation_data=val_ds,epochs=10)

# saving the model
model.save("models/breast_cancer_model.h5")
print("model saved ")

loss,acc=model.evaluate(test_ds)
print(f"accuracy: {acc:.4f}")