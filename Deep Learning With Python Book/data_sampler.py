import os, shutil
from pathlib import Path

# path = 'C:/Users/Luke/Desktop/CodeAcademy Cheat Sheets/Deep Learning With Python'
#
# original_dataset_dir = path + '/kaggle_data/'
#
# base_dir = path + '/cats_dogs_small'
# os.mkdir(base_dir)
#
# # Main Directory Creation
# train_dir = os.path.join(base_dir, 'train')
# os.mkdir(train_dir)
# validation_dir = os.path.join(base_dir, 'validation')
# os.mkdir(validation_dir)
# test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)
#
# # Cat data
# train_cats_dir = os.path.join(train_dir, 'cats')
# os.mkdir(train_cats_dir)
# validation_cats_dir = os.path.join(validation_dir, 'cats')
# os.mkdir(validation_cats_dir)
# test_cats_dir = os.path.join(test_dir, 'cats')
# os.mkdir(test_cats_dir)
#
# # Dog data
# train_dogs_dir = os.path.join(train_dir, 'dogs')
# os.mkdir(train_dogs_dir)
# validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# os.mkdir(validation_dogs_dir)
# test_dogs_dir = os.path.join(test_dir, 'dogs')
# os.mkdir(test_dogs_dir)
#
# # Copy the first 1000 pictures to train_cat
# fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, 'train/' + fname)
#     dst = os.path.join(train_cats_dir, fname)
#     # shutil.copyfile(src, dst)
#
#     print(src)
#     print(dst)


data_path = Path('C:/Users/Luke/Desktop/CodeAcademy Cheat Sheets/Deep Learning With Python/')

original_dataset_dir = data_path / "kaggle_data" / "train"

base_dir = data_path / "cats_dogs_small"
os.mkdir(base_dir)

# Main Directory
train_dir = base_dir / "train"
os.mkdir(train_dir)
validation_dir = base_dir / "validation"
os.mkdir(validation_dir)
test_dir = base_dir / "test"
os.mkdir(test_dir)

# Cat data
train_cats_dir = train_dir / "cats"
os.mkdir(train_cats_dir)
validation_cats_dir = validation_dir / "cats"
os.mkdir(validation_cats_dir)
test_cats_dir = test_dir / "cats"
os.mkdir(test_cats_dir)

# Dog data
train_dogs_dir = train_dir / "dogs"
os.mkdir(train_dogs_dir)
validation_dogs_dir = validation_dir / "dogs"
os.mkdir(validation_dogs_dir)
test_dogs_dir = test_dir / "dogs"
os.mkdir(test_dogs_dir)


# Copy the first 1000 pictures to train_cat
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = original_dataset_dir / fname
    dst = train_cats_dir / fname
    shutil.copyfile(src, dst)

# Copy the next 500 to validation_cat
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = original_dataset_dir / fname
    dst = validation_cats_dir / fname
    shutil.copyfile(src, dst)

# Copy the next 500 to test_cat
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = original_dataset_dir / fname
    dst = test_cats_dir / fname
    shutil.copyfile(src, dst)

# ============================================================================================
# ============================================================================================

# Copy the first 1000 pictures to train_dog
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = original_dataset_dir / fname
    dst = train_dogs_dir / fname
    shutil.copyfile(src, dst)

# Copy the next 500 to validation_dog
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = original_dataset_dir / fname
    dst = validation_dogs_dir / fname
    shutil.copyfile(src, dst)

# Copy the next 500 to test_dog
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = original_dataset_dir / fname
    dst = test_dogs_dir / fname
    shutil.copyfile(src, dst)
