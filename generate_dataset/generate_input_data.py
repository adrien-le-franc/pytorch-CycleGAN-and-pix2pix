import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_img_names_rand(dir_list, N):
    list_img = []

    for dir_ in dir_list:
        print(dir_)
        image_names = os.listdir(dir_)
        index_rand = np.random.permutation(len(image_names))
        for i in range(int(N/len(dir_list))):
            list_img.append(dir_ + image_names[index_rand[i]])
    return list_img

def input_images(images_list, output_file, size_x, size_y, gray=False, edges=False, blur=False, size_blur=5):
    for i in tqdm(range(len(images_list))):
        img_init = cv2.imread(images_list[i])
        img = img_init.copy()
        (size_y_img, size_x_img, _) = np.shape(img)
        #assert(size_y_img <= size_y & size_x_img <= size_x)
        
        if gray:  # Get Grey image
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

        if edges:  # Extract only edges
            edges = cv2.Canny(img, 100, 200)
            img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        if blur:  # Add noise
            img = cv2.blur(img, (size_blur, size_blur))

        img_resized = cv2.copyMakeBorder(img, top=size_y - size_y_img, bottom=0,
                                 left=size_x - size_x_img, right=0,
                                 borderType=cv2.BORDER_CONSTANT)
        img_init_resized = cv2.copyMakeBorder(img_init, top=size_y - size_y_img, bottom=0,
                                 left=size_x - size_x_img, right=0,
                                 borderType=cv2.BORDER_CONSTANT)
        final_img = np.concatenate((img_init_resized, img_resized), axis=1)
        cv2.imwrite(output_file + '/image_test_{}.jpg'.format(i), final_img)

