import tensorflow as tf

img_path = 'D:/kaggle/feiyan/postcrop_train/NORMAL/1.png'

img_raw = tf.io.gfile.GFile(img_path, 'rb').read()

img_tensor = tf.image.decode_image(img_raw)

print(img_tensor.shape)
print(img_tensor.dtype)

'''
根据模型调整大小
'''
img_final = tf.image.resize(img_tensor,[128,128])
img_final = img_final/255.0
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())

'''
把这些包装再一个简单的函数里
'''


def preprocess_image(image):
    '''
    image=img_raw
    '''
    image = tf.image.decode_png(image, channels=3)

    image = tf.image.resize(image, [128, 128])
    image /= 255.0  # 归一化
    return image


def load_and_preprocess_image(path):
    '''
    param:img_path
    return:img_raw
    '''
    image = tf.io.gfile.GFile(img_path, 'rb').read()
    return preprocess_image(image)

'''
所有图片的path

list
'''
import os

all_image_paths = os.listdir('D:/kaggle/feiyan/postcrop_train/NORMAL/')
all_image_paths = all_image_paths+(os.listdir('D:/kaggle/feiyan/postcrop_train/PNEUMONIA/'))

print(len(all_image_paths))

#from_tensor_slices:将字符串数组切片，得到一个字符串数据集
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
print(path_ds)

'''
来动态加载每一个图片进入函数load_and_preprocess_image
'''
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

'''
添加一个标签数据集
'''
label_names = {0: 'NORMAL', 1: 'PNEUMONIA'}
all_image_labels = []
for i in range(1341):
    all_image_labels.append(0)
for i in range(1341, 5216):
    all_image_labels.append(1)

print(len(all_image_labels))
# print(all_image_labels[1340],all_image_labels[1341])

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
print(label_ds)
for label in label_ds.take(10):
    print(label_names[label.numpy()])

'''
image_ds 和 label_ds打包成一个(图片，标签)数据集
格式：ZipDataset
'''
image_label_ds = tf.data.Dataset.zip((image_ds,label_ds))
print(image_label_ds)

'''
1.打乱数据
2.分割数据
3.重复数据
4.在处理当前元素时准备后面的元素
'''
BATCH_SIZE = 32
image_count = 5216
# 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
# 被充分打乱。
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
#这允许在处理当前元素时准备后面的元素
ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
ds

from tensorflow.keras import datasets, layers, models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))
a
# model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(ds, epochs=10,steps_per_epoch=12)

