import cv2
import numpy as np
import os
#import matplotlib.pyplot as plt

def compute_dist_img(img1, img2):
    diff = cv2.addWeighted(img1, -1, img2, 1, 0)
    return np.linalg.norm(diff)

def plot_img(img):
    plt.figure()
    plt.imshow(img)
    plt.plot()

def generate_IP(image_name_list, N, output_file, seuil_min=3500, seuil_max=40000, max_im=3, save_test=False):
    index_rand = np.random.permutation(len(image_name_list))

    n_iter = 0
    nb_input = 0
    while nb_input < N:
        i = index_rand[n_iter]
        img1 = cv2.imread(image_name_list[i])
        img2 = cv2.imread(image_name_list[i+1])
        dist_12 = compute_dist_img(img1, img2)

        print(dist_12)

        nb_img2 = -1
        while dist_12 < seuil_min and nb_img2 < max_im:
            nb_img2 += 1
            img2 = cv2.imread(image_name_list[i + 1 + nb_img2 + 1])
            dist_12 = compute_dist_img(img1, img2)

        if nb_img2 != max_im and dist_12 < seuil_max:
            img3 = cv2.imread(image_name_list[i + 2 + nb_img2 + 1])
            dist_23 = compute_dist_img(img2, img3)

            nb_img3 = -1
            while dist_23 < seuil_min and nb_img3 < max_im:
                nb_img3 += 1
                img2 = cv2.imread(image_name_list[i + 3 + nb_img2 + nb_img3 + 1])
                dist_23 = compute_dist_img(img2, img3)

            if nb_img3 != max_im and dist_23 < seuil_max:
                img_interp = cv2.addWeighted(img1, 0.5, img3, 0.5, 0)

                nb_input += 1
                final_img = np.concatenate((img_interp, img2), axis=1)
                cv2.imwrite(output_file + '/input_IP_{}.jpg'.format(nb_input), final_img)

                if save_test:
                    test = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
                    test2 = cv2.addWeighted(img2, 0.5, img3, 0.5, 0)
                    test_concat = np.concatenate((test, test2), axis=1)
                    cv2.imwrite(output_file + '/test_input_IP_{}.jpg'.format(nb_input), test_concat)

        n_iter += 1
        print(nb_input)

if __name__ == '__main__':
    dir_ = "../datasets/S02E28/"
    image_name_list = [dir_ + name for name in os.listdir(dir_)]

    generate_IP(image_name_list, N=100, output_file='../datasets/input_IP',
                seuil_min=2000, seuil_max=40000,
                max_im=4, save_test=False)
