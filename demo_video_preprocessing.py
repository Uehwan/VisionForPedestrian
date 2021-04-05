import os
import glob
import ntpath
import argparse

from lib.utils.demo_utils import video_to_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str,
                        help='root folder containing videos')
    parser.add_argument('--output_folder', type=str,
                        help='output folder to save results')
    args = parser.parse_args()

    sub_dirs = os.listdir(args.root_folder)
    for sub_dir in sub_dirs:
        video_files = glob.glob(os.path.join(args.root_folder, sub_dir, "*.mp4"))
        for video_file in video_files:
            target_name = ntpath.basename(video_file)[:-4]
            target_folder = os.path.join(args.output_folder, sub_dir, target_name)
            video_to_images(video_file, target_folder)
        