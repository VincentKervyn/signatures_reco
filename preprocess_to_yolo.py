import os
import shutil
import cv2
import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, model_selection

###
df = []
# refer the xml files to understand its structure and revisit this code block.
annotations = sorted(glob('data/train_xml/*.xml'))
for file in annotations:
    myroot = ET.parse(file).getroot()
    f_name = os.path.splitext(file)
    page_height, page_width = myroot[0][0].attrib['height'], myroot[0][0].attrib['width']
    name = str(f_name[0]).replace("data/train_xml/", "") + '.tif'
    row = []
    # An image might have multiple items (zones) (logos and signs), so iterate through each zones
    for zone in myroot[0][0]:
        category = zone.attrib['gedi_type']  # type of zone (DLLogo/ DLSignature)
        id = zone.attrib['id']
        x, y = zone.attrib['col'], zone.attrib['row']  # x, y coordinate
        w, h = zone.attrib['width'], zone.attrib['height']  # width and height of bbox

        # Signature have Authors, representing whose signature it is
        if category == 'DLSignature':
            AuthorID = zone.attrib['AuthorID']
            Overlapped = zone.attrib['Overlapped']
        else:
            # Logos don't have authors.
            AuthorID, Overlapped = ('NA', 'NA')
        row = [name, page_height, page_width, AuthorID, Overlapped, category, id, x, y, w, h]
        df.append(row)
##Save data in DataFrame
data = pd.DataFrame(df, columns=['name', 'page_height', 'page_width', 'AuthorID', 'Overlapped',
                                 'category', 'id', 'x', 'y', 'width', 'height'])
test = data[['page_height', 'page_width']]
# print (data)
# print(test.max(),test.min())

## Scaling the image to reduce training time:
# To save on training time, resize the images to a maximum height and width of 640 and 480. While resizing the image,
# the bounding box coordinates also changes. This code computes how much each image is shrunken and updates
# the bounding box coordinates appropriately.

BASE_DIR = 'data/train/'
SAVE_PATH = 'data/scaled/'


# os.mkdir(SAVE_PATH)


def scale_image(data):
    df_new = []
    filename = data.name
    X, Y, W, H = map(int, data.x), map(int, data.y), map(int, data.width), map(int, data.height)
    for file, x, y, w, h in zip(filename, X, Y, W, H):
        image_path = BASE_DIR + file
        # print(f'image path: {image_path}')
        img = cv2.imread(image_path, 1)
        page_height, page_width = img.shape[:2]
        max_height = 640
        max_width = 480

        # computes the scaling factor
        if max_height < page_height or max_width < page_width:
            scaling_factor = max_height / float(page_height)
            if max_width / float(page_width) < scaling_factor:
                scaling_factor = max_width / float(page_width)
            # scale the image with the scaling factor
            img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        jpg_filename = file[:-4] + '.jpg'
        new_file_path = SAVE_PATH + jpg_filename
        cv2.imwrite(new_file_path, img)  # write the scales image

        # save new page height and width
        page_height, page_width = page_height * scaling_factor, page_width * scaling_factor
        # compute new x, y, w, h coordinates after scaling
        x, y, w, h = int(x * scaling_factor), int(y * scaling_factor), int(w * scaling_factor), int(h * scaling_factor)
        row = [jpg_filename, x, y, w, h, page_height, page_width]
        df_new.append(row)
    return df_new


scaled_data = scale_image(data)

scaled_data = list(zip(*scaled_data))

data['name'] = scaled_data[0]
data['x_scaled'] = scaled_data[1]
data['y_scaled'] = scaled_data[2]
data['w_scaled'] = scaled_data[3]
data['h_scaled'] = scaled_data[4]
data['page_height_scaled'] = scaled_data[5]
data['page_width_scaled'] = scaled_data[6]

########################
# Let's try if it working
img = cv2.imread('data/scaled/00060f1db73c837c2b943fec640f920a_1.jpg')

img = cv2.rectangle(img, (244, 338), (244 + 107, 338 + 30), (255, 0, 0), 1)
# (x_scaled, y_scaled), (x_scaled + width_scaled, y_scaled + h_scaled)
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.show()

###Saving to CSV file


data.to_csv('tobacco_cleaned.csv', index=False)


### Converting to yolo format
def x_center(df):
    return int(df.x_scaled + (df.w_scaled / 2))


def y_center(df):
    return int(df.y_scaled + (df.h_scaled / 2))


def w_norm(df, col):
    return df[col] / df['page_width_scaled']


def h_norm(df, col):
    return df[col] / df['page_height_scaled']


df = pd.read_csv('tobacco_cleaned.csv')

le = preprocessing.LabelEncoder()
le.fit(df['category'])
print(le.classes_)
labels = le.transform(df['category'])
df['labels'] = labels

df['x_center'] = df.apply(x_center, axis=1)
df['y_center'] = df.apply(y_center, axis=1)

df['x_center_norm'] = df.apply(w_norm, col='x_center', axis=1)
df['width_norm'] = df.apply(w_norm, col='w_scaled', axis=1)

df['y_center_norm'] = df.apply(h_norm, col='y_center', axis=1)
df['height_norm'] = df.apply(h_norm, col='h_scaled', axis=1)

print(df.head(1))

## Moving images to train and valid folders

df_train, df_valid = model_selection.train_test_split(df, test_size=0.1, random_state=13, shuffle=True)
print(df_train.shape, df_valid.shape)

## To create folder
# os.mkdir('data/tobacco_yolo_format')
# os.mkdir('data/tobacco_yolo_format/images')
# os.mkdir('data/tobacco_yolo_format/images/train')
# os.mkdir('data/tobacco_yolo_format/images/valid')
#
# os.mkdir('data/tobacco_yolo_format/labels')
# os.mkdir('data/tobacco_yolo_format/labels/train')
# os.mkdir('data/tobacco_yolo_format/labels/valid')

##Segregating images and labels to train and valid
def segregate_data(df, img_path, label_path, train_img_path, train_label_path):
    # global row
    filenames = []
    for name in df.name:
        filenames.append(name)
    filenames = set(filenames)

    for name in filenames:
        yolo_list = []

        for _, row in df[df.name == name].iterrows():
            yolo_list.append([row.labels, row.x_center_norm, row.y_center_norm, row.width_norm, row.height_norm])
        print(name)
        print(row.name)
        yolo_list = np.array(yolo_list)
        txt_filename = os.path.join(train_label_path, str(row.name).replace(".jpg", "")) + '.txt'

        print('txt filename is ', txt_filename)
        # Save the .img & .txt files to the corresponding train and validation folders
        np.savetxt(txt_filename, yolo_list, fmt=["%d", "%f", "%f", "%f", "%f"])
        shutil.copyfile(os.path.join(img_path, str(name)), os.path.join(train_img_path, str(row.name) + '.jpg'))


# Apply function
src_img_path = "data/scaled/"
src_label_path = "data/groundtruth/"

train_img_path = "data/tobacco_yolo_format/images/train"
train_label_path = "data/tobacco_yolo_format/labels/train"

valid_img_path = "data/tobacco_yolo_format/images/valid"
valid_label_path = "data/tobacco_yolo_format/labels/valid"

segregate_data(df_train, src_img_path, src_label_path, train_img_path, train_label_path)
segregate_data(df_valid, src_img_path, src_label_path, valid_img_path, valid_label_path)

print("No. of Training images", len(os.listdir('data/tobacco_yolo_format/images/train')))
print("No. of Training labels", len(os.listdir('data/tobacco_yolo_format/labels/train')))

print("No. of valid images", len(os.listdir('data/tobacco_yolo_format/images/valid')))
print("No. of valid labels", len(os.listdir('data/tobacco_yolo_format/labels/valid')))
