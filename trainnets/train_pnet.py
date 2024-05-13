import argparse
from train_net import Trainer

parser = argparse.ArgumentParser(
    description='Generate training data for pnet.')
parser.add_argument('-e', dest='epoch', type=int)
parser.add_argument('-b', dest='batch_size', type=int)
parser.add_argument('-o', dest="output_filename", help="Path to save the model.")
parser.add_argument('-d', dest="data_train", default="../output/data_train", type=str, help="Folder that save training data for pnet.")
parser.add_argument('-dv', dest="device", default='cpu', type=str, help="'gpu', 'cuda:0' and so on.")
parser.add_argument('-w',dest = "workers", default = 1, type = int, help = "number of dataloader workers")
parser.add_argument('-prof',dest = "prof", default = 0, type = int, help = "Enable/Disable profiling")
parser.add_argument('-wandb', dest = "wandb", default = 0,type = int, help = 'Enable/Disable for WandB logging')

args = parser.parse_args()

trainer = Trainer('pnet', device=args.device, log_dir='./runs/pnet/',n_workers = args.workers,prof = args.prof,usewandb = args.wandb)
trainer.train(args.epoch, args.batch_size, args.data_train)
trainer.export_model(args.output_filename)
