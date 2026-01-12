import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# """
# The structure for navigating the dataset images are as follows
# Filepath | Category | Class
#
# Filepath  : The path of file the image is
# Category  : Training/ Testing data
# Class     : glioma/ meningioma/ notumor/ pituitary
# """

rows = []

for category in ['Training', 'Testing']:
    category_path = os.path.join("data", category)

    for class_name in os.listdir(category_path):
        class_path = os.path.join(category_path, class_name)

        if os.path.isdir(class_path):
            for filename in tqdm(os.listdir(class_path), desc=f"{category}/{class_name}", unit="file", dynamic_ncols=True):
                if filename.lower().endswith(('.png', '.jpg', 'jpeg')):
                    filepath = os.path.join(f"../../data/{category}/{class_name}/", filename)

                    rows.append((filepath, category, class_name))

directory_df = pd.DataFrame(rows)
directory_df.columns = ['filepath', 'category', 'class_name']

directory_df.to_csv("data/csv/directory_dataset.csv", index=False)

print(f"Directory dataframe is created")

train_df = directory_df[directory_df['category'] == 'Training'].reset_index(drop=True)
test_df = directory_df[directory_df['category'] == 'Testing'].reset_index(drop=True)

train_df.to_csv('data/csv/train.csv', index=False)
test_df.to_csv('data/csv/test.csv', index=False)

print(f"Splitting the training and testing is done!")