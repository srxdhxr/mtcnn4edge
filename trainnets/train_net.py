import os
import torch
import glob
import progressbar
import sys
sys.path.append('../')
import time
from model.mtcnn_pytorch import PNet, RNet
from datagen.data import MtcnnDataset
from torch.utils.tensorboard import SummaryWriter

class Trainer(object):

    def __init__(self, net_stage, device='cpu', log_dir='./runs', output_folder='./runs', resume=False,n_workers = 1,prof = False):
        
        self.net_stage = net_stage
        self.device = device
        self.output_folder = output_folder
        self.prof = prof
        if net_stage == 'pnet':
            self.net = PNet(is_train=True, device=self.device)
        
        elif net_stage == 'rnet':
            self.net = RNet(is_train=True, device=self.device)
        
        
        self.optimizer = torch.optim.Adam(self.net.parameters())
        
        self.n_workers = n_workers
        self.globle_step = 1
        self.epoch_num = 1

        # if resume:
        #     self.load_state_dict()
        
        self.writer = SummaryWriter(log_dir=log_dir, purge_step=self.epoch_num)
        
    def train(self, num_epoch, batch_size, data_folder):
        dataset = MtcnnDataset(data_folder, self.net_stage, batch_size, suffix=self.net_stage,num_workers =self.n_workers)
        eval_dataset = MtcnnDataset(data_folder, self.net_stage, batch_size, suffix=self.net_stage+'_eval',num_workers =self.n_workers)

        for i in range(num_epoch - self.epoch_num + 1):
            print("Training epoch %d ......" % self.epoch_num)
            data_iter, total_batch = dataset.get_iter()
            self._train_epoch(data_iter, total_batch)
            print("Training epoch %d done." % self.epoch_num)

            if(self.prof == 0):
                print("Evaluate on training data...")
                data_iter, total_batch = dataset.get_iter()
                train_result = self.eval(data_iter, total_batch)
                self.writer.add_scalars(f"accuracy/{self.net_stage}", {"train":train_result['accuracy']}, global_step=self.epoch_num)


                print("Evaluate on eval data...")
                data_iter, total_batch = eval_dataset.get_iter()
                eval_result = self.eval(data_iter, total_batch)

                self.writer.add_scalars(f"accuracy/{self.net_stage}", {"validation":eval_result['accuracy']}, global_step=self.epoch_num)



            self.save_state_dict()

            self.epoch_num += 1

    def _train_epoch(self, data_iter, total_batch):
        
        bar = progressbar.ProgressBar(max_value=total_batch)
        total_dl_time = 0
        total_tr_time = 0

        for i, batch in enumerate(data_iter):
            bar.update(i)

            loss,dl_time,tr_time = self._train_batch(batch)

            #Accumulate Dataloading Time and Training Time
            total_dl_time += dl_time
            total_tr_time += tr_time

            self.writer.add_scalar(f'train/{self.net_stage}/batch_loss', loss, global_step=self.globle_step)
            self.globle_step += 1

        bar.update(total_batch)    
        self.writer.add_scalars(f'Dataloader/{self.net_stage}',{f"worker_{self.n_workers}":total_dl_time}, global_step=self.epoch_num)
        self.writer.add_scalar(f'train/{self.net_stage}/Average Train Time (Epoch)', total_tr_time, global_step=self.epoch_num)

    def _train_batch(self, batch):

        # assemble batch
        dl_start_time = time.time()
        images, labels, boxes_reg = self._assemble_batch(batch)
        dl_time = time.time() - dl_start_time
        # train step

        tr_start_time = time.time()
        self.optimizer.zero_grad()
        loss = self.net.get_loss(images, labels, boxes_reg)
        loss.backward()
        self.optimizer.step()
        tr_time = time.time()-tr_start_time


        return loss,dl_time,tr_time


    def _assemble_batch(self, batch):
        # assemble batch
        (pos_img, pos_reg), (part_img, part_reg), (neg_img, neg_reg) = batch

        # stack all images together
        images = torch.cat([pos_img, part_img, neg_img]).to(self.device)

        # create labels for each image. 0 (neg), 1 (pos), 2 (part)
        pos_label = torch.ones(pos_img.shape[0], dtype=torch.long)
        part_label = torch.ones(part_img.shape[0], dtype=torch.long) * 2
        neg_label = torch.zeros(neg_img.shape[0], dtype=torch.long)

        labels = torch.cat([pos_label, part_label, neg_label]).to(self.device)

        # stack boxes reg
        boxes_reg = torch.cat([pos_reg, part_reg, neg_reg]).to(self.device)

        
        return images, labels, boxes_reg
    
    def eval(self, data_iter, total_batch):
        total = 0
        right = 0
        tp = 0  # True positive
        fp = 0  # False positive
        fn = 0  # False negative
        tn = 0  # True negative

        total_cls_loss = 0
        total_box_loss = 0

        bar = progressbar.ProgressBar(max_value=total_batch)

        for i, batch in enumerate(data_iter):
            bar.update(i)

            # assemble batch
            images, gt_label, gt_boxes = self._assemble_batch(batch)
            
            # Forward pass
            with torch.no_grad():
                pred_label, pred_offset = self.net.forward(images)

            # Reshape the tensor
            pred_label = pred_label.view(-1, 2)
            pred_offset = pred_offset.view(-1, 4)

            # Compute the loss
            total_cls_loss += self.net.cls_loss(gt_label, pred_label)
            total_box_loss += self.net.box_loss(gt_label, gt_boxes, pred_offset)
          

            # compute the classification acc
            pred_label = torch.argmax(pred_label, dim=1)

            mask = gt_label <= 1
            right += torch.sum(gt_label[mask] == pred_label[mask])
            total += gt_label[mask].shape[0]
        
            p_mask = gt_label == 1
            tp += torch.sum(gt_label[p_mask] == pred_label[p_mask])
            fp += torch.sum(gt_label[p_mask] != pred_label[p_mask])

            n_mask = gt_label == 0
            tn += torch.sum(gt_label[n_mask] == pred_label[n_mask])
            fn += torch.sum(gt_label[n_mask] != pred_label[n_mask])

        bar.update(total_batch)

        acc = right.float() / total
        precision = tp.float() / (tp + fp)
        recall = tp.float() / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        avg_cls_loss = total_cls_loss / i
        avg_box_loss = total_box_loss / i
        avg_loss = (avg_cls_loss+avg_box_loss)/2
        return {"accuracy":acc, "avg_loss":  avg_loss}


    def save_state_dict(self):
        checkpoint_name = "checkpoint_epoch_%d" % self.epoch_num
        file_path = os.path.join(self.output_folder, checkpoint_name)

        state = {
            'epoch_num': self.epoch_num,
            'state_dict': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, file_path)

    def export_model(self, filename):
        torch.save(self.net.state_dict(), filename)

    def load_state_dict(self):

        # Get the latest checkpoint in output_folder
        all_checkpoints = glob.glob(os.path.join(self.output_folder, 'checkpoint_epoch_*'))

        if len(all_checkpoints) > 1:
            epoch_nums = [int(i.split('_')[-1]) for i in all_checkpoints]
            max_index = epoch_nums.index(max(epoch_nums))
            latest_checkpoint = all_checkpoints[max_index]

            state = torch.load(latest_checkpoint)
            self.epoch_num = state['epoch_num'] + 1
            self.net.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer']) 
