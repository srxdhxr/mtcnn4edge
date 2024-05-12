import argparse
import torch
from wider_face import *

from gen_onet_train import generate_training_data_for_onet

from model.mtcnn_pytorch import PNet,RNet


parser = argparse.ArgumentParser(
    description='Generate training data for onet.')
parser.add_argument('-pm', type=str, dest="pnet_model_file", help="Pre-trained pnet model file.")
parser.add_argument('-rm', type=str, dest="rnet_model_file", help="Pre-trained rnet model file.")
parser.add_argument('-o', dest="output_folder", default="../output/data_train", type=str, help="Folder to save training data for onet.")
parser.add_argument("-d", dest="detection_dataset",type=str, default="WiderFace",
                    help="Face Detection dataset name.")

args = parser.parse_args()


# load pre-trained pnet
print("Loading pre-trained pnet.")
device = 'cuda'
pnet = PNet(device=device, is_train=False)
pnet.load(args.pnet_model_file)
rnet = RNet(device=device, is_train=False)
rnet.load(args.rnet_model_file)

detection_dataset = WiderFace(dataset_folder="/home/srxdhxr/WIDER_FACE")
detection_meta = detection_dataset.get_train_meta()
detection_eval_meta = detection_dataset.get_val_meta()
print("Start generate classification and bounding box regression training data.")
#generate_training_data_for_onet(pnet, rnet, detection_meta, args.output_folder, crop_size=48, suffix='onet')
print("Done")

print("Start generate classification and bounding box regression eval data.")
generate_training_data_for_onet(pnet, rnet, detection_eval_meta, args.output_folder, crop_size=48, suffix='onet_eval')
print("Done")
