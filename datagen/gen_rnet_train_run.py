import argparse
import torch

from gen_rnet_train import generate_training_data_for_rnet
from model.mtcnn_pytorch import PNet
from wider_face import *


parser = argparse.ArgumentParser(
    description='Generate training data for rnet.')
parser.add_argument('-m', type=str, dest="model_file", help="Pre-trained model file.")
parser.add_argument('-o', dest="output_folder", default="../output/data_train", type=str, help="Folder to save training data for rnet.")
parser.add_argument("-d", dest="detection_dataset",type=str, default="WiderFace",
                    help="Face Detection dataset name.")
parser.add_argument("-l", type=str, dest="landmarks_dataset", default="CelebA",
                    help="Landmark localization dataset name.")
args = parser.parse_args()

# load pre-trained pnet
print("Loading pre-trained pnet.")
device = 'cpu'
pnet = PNet(device=device, is_train=False)
pnet.load(args.model_file)


detection_dataset = WiderFace(dataset_folder="/home/srxdhxr/WIDER_FACE")
detection_meta = detection_dataset.get_train_meta()
detection_eval_meta = detection_dataset.get_val_meta()
print("Start generate classification and bounding box regression training data.")
generate_training_data_for_rnet(pnet, detection_meta, args.output_folder, crop_size=24, suffix='rnet')
print("Done")

print("Start generate classification and bounding box regression eval data.")
generate_training_data_for_rnet(pnet, detection_eval_meta, args.output_folder, crop_size=24, suffix='rnet_eval')
print("Done")
