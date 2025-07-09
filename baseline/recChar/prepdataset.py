import os
from PIL import Image
from tqdm import tqdm
import random

def extract_bbox_and_label(filename):
    parts = filename.replace('.jpg', '').split('-')
    bbox = parts[2]
    label_code = parts[4]

    x1, y1 = map(int, bbox.split('_')[0].split('&'))
    x2, y2 = map(int, bbox.split('_')[1].split('&'))

    label_parts = label_code.split('_')
    indices = list(map(int, label_parts))

    provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
                 "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
    alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
                 'X', 'Y', 'Z', 'O']
    ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
           'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

    label_chars = [provinces[indices[0]], alphabets[indices[1]]]
    for i in range(2, 7):
        label_chars.append(ads[indices[i]])

    label = ''.join(label_chars)
    return (x1, y1, x2, y2), label

def prepare_ccpd_dataset(ccpd_dir, output_dir, max_images=None, train_split=0.5):
    os.makedirs(output_dir, exist_ok=True)
    train_img_dir = os.path.join(output_dir, 'train', 'images')
    val_img_dir = os.path.join(output_dir, 'val', 'images')
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)

    train_label_path = os.path.join(output_dir, 'train', 'labels.txt')
    val_label_path = os.path.join(output_dir, 'val', 'labels.txt')

    all_images = [f for f in os.listdir(ccpd_dir) if f.endswith('.jpg')]
    random.shuffle(all_images)

    if max_images is not None:
        selected_images = all_images[:max_images]
    else:
        selected_images = all_images

    split_index = int(len(selected_images) * train_split)
    train_images = selected_images[:split_index]
    val_images = selected_images[split_index:]

    def save_subset(images, img_dir, label_file):
        with open(label_file, 'w', encoding='utf-8') as f:
            for idx, img_name in enumerate(tqdm(images)):
                bbox, label = extract_bbox_and_label(img_name)
                img_path = os.path.join(ccpd_dir, img_name)
                img = Image.open(img_path)
                crop = img.crop(bbox)
                new_img_name = f'{idx}.jpg'
                crop.save(os.path.join(img_dir, new_img_name))
                f.write(f'{new_img_name}\t{label}\n')

    print("Preparing training set...")
    save_subset(train_images, train_img_dir, train_label_path)
    print("Preparing validation set...")
    save_subset(val_images, val_img_dir, val_label_path)
    print("Dataset preparation complete!")

if __name__ == '__main__':
    ccpd_dir = '../Downloads/CCPD2019/CCPD2019/ccpd_base'
    output_dir = './ccpd_dataset'
    prepare_ccpd_dataset(ccpd_dir, output_dir, max_images=None, train_split=0.5)
