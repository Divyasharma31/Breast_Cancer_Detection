import tensorflow as tf
import os

def get_image_path_and_labels(data_root):
    image_paths=[]
    labels=[]
    for folder in os.listdir(data_root):
        folder_path=os.path.join(data_root,folder)
        if os.path.isdir(folder_path):
            for label_dir in ['0','1']:
                class_path=os.path.join(folder_path,label_dir)
                if os.path.isdir(class_path):
                    for file_name in os.listdir(class_path):
                        if file_name.endswith('.png'):
                            image_paths.append(os.path.join(class_path,file_name))
                            labels.append(int(label_dir))
    return image_paths,labels

def preprocess_img(image_path,label,img_size=(150,150)):
    image=tf.io.read_file(image_path)
    image=tf.image.decode_png(image,channels=3)
    image=tf.image.resize(image,img_size)
    image=image/255.0 #normalize to [0,1]
    return image,label

def create_dataset(data_root,batch_size=32,shuffle=True,buffer_size=10000,img_size=(150,150)):
    image_paths,labels=get_image_path_and_labels(data_root)
    dataset=tf.data.Dataset.from_tensor_slices((image_paths,labels))
    dataset=dataset.map(lambda x,y:preprocess_img(x,y,img_size),num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset=dataset.shuffle(buffer_size)
    dataset=dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


