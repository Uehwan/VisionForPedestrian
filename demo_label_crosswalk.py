import cv2
import os
import glob
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle


def onclick(event):
    ix, iy = event.xdata, event.ydata
    print('x = {}, y = {}'.format(ix, iy))

    global coords
    coords.append((ix, iy))

    return coords

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        help='the path to root dir')
    args = parser.parse_args()    
    
    PATH_IMAGE = args.root_dir

    annotation_results = {}
    dirs = os.listdir(PATH_IMAGE)
    for i, dir_one in enumerate(sorted(dirs)):
        coords = []
        image_files = sorted(glob.glob(os.path.join(PATH_IMAGE, dir_one, "*.jpg")))
        plt.close('all')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        image_one = cv2.imread(image_files[random.choice(range(len(image_files)))])
        ax.imshow(image_one)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        annotation_results[dir_one] = coords

    with open('anno_crosswalk.pickle', 'wb') as pickle_file:
        pickle.dump(annotation_results, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open('anno_crosswalk.pickle', 'rb') as pickle_file:
        result = pickle.load(pickle_file)
