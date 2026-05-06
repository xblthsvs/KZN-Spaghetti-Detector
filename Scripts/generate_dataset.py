import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image

csv_table = "/Users/svstemp/Documents/GitHub/KZN-Spaghetti-Detector/Training Data/Printing_Errors 2/general_data/black_bed_all.csv"
images_folder = "/Users/svstemp/Documents/GitHub/KZN-Spaghetti-Detector/Training Data/Printing_Errors 2/images/all_images256/"

def load_image(path, size=(256, 256)):
    with Image.open(path) as img:
        return np.array(img.convert("RGB").resize(size))

columns_to_keep = ['image', 'class']
df = pd.read_csv(csv_table, sep=';', usecols=columns_to_keep)
df['split'] = pd.Series()

train_df, test_df = train_test_split(df, test_size=0.2, random_state=1)
train_df["split"] = "train"
test_df["split"] = "test"

df = pd.concat([train_df, test_df])
df = df.sort_index()

x_train = []
x_test = []
y_train = []
y_test = []

for i, row in df.iterrows():
    image_path = images_folder + row['image']
    label = row['class']
    
    image_array = load_image(image_path)
    
    if row['split'] == 'train':
        x_train.append(image_array)
        y_train.append(label)
    else:
        x_test.append(image_array)
        y_test.append(label)


x_train = np.stack(x_train)
x_test = np.stack(x_test)
y_train = np.stack(y_train)
y_test = np.stack(y_test)

np.savez("/Users/svstemp/Documents/GitHub/KZN-Spaghetti-Detector/Training Data/black_bed_all.npz",
         x_train=x_train,
         x_test=x_test,
         y_train=y_train,
         y_test=y_test)
         
