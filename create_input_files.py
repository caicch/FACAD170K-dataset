#!/usr/bin/env python3

from utils import create_input_files
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image_Captioning')
    parser.add_argument('--dataset', default="facad", help='Default MSCOCO 14 Dataset.')
    parser.add_argument('--json_path', default="./dataset/caption_dataset/FACAD_13k1.json",
                        help='path of captions dataset.')
    parser.add_argument('--image_folder', default="/home/chen/Desktop/Image-Caption-master/dataset", help='path of image dataset.')
    parser.add_argument('--captions_per_image', type=int, default=1, help='How many captions each image has?')
    parser.add_argument('--min_word_freq', type=int, default=5, help='the minimum frequency of words')
    parser.add_argument('--output_folder', default='/home/chen/Desktop/Image-Caption-master/dataset/generated_facad13', help='output filepath.')
    parser.add_argument('--max_len', type=int, default=30, help='the maximum length of each caption.')
    args = parser.parse_args()

    if not (os.path.exists(args.output_folder) and os.path.isdir(args.output_folder)):
        os.makedirs(args.output_folder)
    # Create input files (along with word map)
    create_input_files(dataset=args.dataset,
                       karpathy_json_path=args.json_path,
                       image_folder=args.image_folder,
                       captions_per_image=args.captions_per_image,
                       min_word_freq=args.min_word_freq,
                       output_folder=args.output_folder,
                       max_len=args.max_len)
