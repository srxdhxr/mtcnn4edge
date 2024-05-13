import argparse
from wider_face import *
from gen_pnet_train import generate_training_data_for_pnet



parser = argparse.ArgumentParser(
    description='Generate training data for pnet.')
parser.add_argument('-o', dest="output_folder", default="../output/data_train", type=str, help="Folder to save training data for pnet.")
parser.add_argument("-d", dest="detection_dataset",type=str, default="WiderFace",
                    help="Face Detection dataset name.")
args = parser.parse_args()

detection_dataset = WiderFace(dataset_folder="/home/srxdhxr/WIDER_FACE")
detection_meta = detection_dataset.get_train_meta()
detection_eval_meta = detection_dataset.get_val_meta()
print("Start generate classification and bounding box regression training data.")
generate_training_data_for_pnet(detection_meta, output_folder=args.output_folder, suffix='pnet')
print("Done")

print("Start generate classification and bounding box regression eval data.")
generate_training_data_for_pnet(detection_eval_meta, output_folder=args.output_folder, suffix='pnet_eval')
print("Done")
