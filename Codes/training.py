# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 18:40:55 2026

@author: Omar Al-maqtari
"""

import os
import time
import copy
import random
from tqdm import tqdm
from datetime import datetime
from dataloader import get_loader

import torch
from torch import nn

from learning import Learner
from model import DINOv3_distill
from utils1 import initialize_model, displayfigures, PRC

import csv
from evaluation import Evaluation
from loss import DistillLoss, DiceLoss, FocalLoss


class Trainer(object):
    def __init__(self, cfg):
        # Config
        self.cfg = cfg

        # Paths
        self.result_path = cfg.result_path
        self.sr_path = cfg.sr_path
        self.net_path = os.path.join(cfg.model_path, cfg.report_name + '.pth')
        
        # Report file
        self.report_name = cfg.report_name
        self.report = open(self.result_path + self.report_name + '.txt', 'a+')
        self.report.write('\n' + str(datetime.now()))
        self.report.write('\n' + str(cfg))

        # Data loader
        self.train_loader = get_loader(cfg, mode='train', aug_prob=cfg.aug_prob)
        self.valid_loader = get_loader(cfg, mode='val', aug_prob=cfg.aug_prob)
        self.test_loader = get_loader(cfg, mode='val', aug_prob=0.)
        
        # Hyper-parameters
        self.lr = 0.
        self.aug_prob = cfg.aug_prob
        self.epochs = cfg.epochs
        self.steps = len(self.train_loader)

        # Models
        print("initialize model...")
        self.experiment_num = cfg.experiment_num
        self.learning_mode = cfg.learning_mode
        self.teacher_type = cfg.teacher_type
        self.model_type = cfg.model_type
        self.optimizer_type = cfg.optimizer_type
        self.lr_sch_type = cfg.lr_sch_type
        self.dataset = cfg.dataset
        self.teacher_type = cfg.teacher_type
        self.loss_eq = []
        self.loss_eq.append(DistillLoss())
        self.loss_eq.append(FocalLoss())
        self.loss_eq.append(DiceLoss())
        self.loss_eq.append(nn.BCELoss())
        
        self.init_model = initialize_model(self.cfg)
        self.model = self.init_model.get_model(self.model_type)
        self.teacher = None
        if self.learning_mode == 'Dist':
            if self.teacher_type == 'large':
                self.teacher = DINOv3_distill(cfg, self.teacher_type, torch.float32)
            elif self.teacher_type == '7B':
                self.teacher = DINOv3_distill(cfg, self.teacher_type, torch.bfloat16)
        elif self.learning_mode == 'Seed':
            self.pretrained_net_path = os.path.join(cfg.model_path,
                                                    self.model_type + '_' + self.dataset + '_' + self.teacher_type
                                                    + '_Dist_'+ str(self.experiment_num) +'.pth')
            self.model.load_state_dict(torch.load(self.pretrained_net_path, map_location='cpu', weights_only=True),
                                        strict=False)

        elif self.learning_mode == 'Seg':
            self.pretrained_net_path = os.path.join(cfg.model_path,
                                                    self.model_type + '_' + self.dataset + '_' + self.teacher_type
                                                    + '_Seed_' + str(self.experiment_num) + '.pth')
            self.model.load_state_dict(torch.load(self.pretrained_net_path, map_location="cpu", weights_only=True),
                                       strict=False)


            self.teacher = copy.deepcopy(self.model)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
            for p in self.model.E.parameters():
                p.requires_grad = False
            self.init_model.model = self.model

        self.optimizer, self.lr_sch = self.init_model.get_optimizer(self.steps)
        self.grad_scaler = torch.amp.GradScaler(enabled=True)
        
        if self.cfg.mode == 'train':
            self.params = 0
            for p in self.model.parameters():
                self.params += p.numel()
            self.cfg.parameters = self.params
            
            print(self.model_type)
            self.report.write('\n' + str(self.model_type))
            print("The number of parameters: {}".format(self.params))
            self.report.write("\n The number of parameters: {}".format(self.params))
            self.report.write('\n' + str(self.model))

    def _lr_state(self, optimizer):
        state = {"muon": [], "adam": []}
        for g in optimizer.param_groups:
            key = "muon" if g.get("use_muon", False) else "adam"
            state[key].append(float(g["lr"]))
        return {k: v for k, v in state.items() if len(v) > 0}

    def _lr_string(self, optimizer):
        state = self._lr_state(optimizer)
        parts = []
        if "muon" in state:
            parts.append("muon=" + "/".join([f"{lr:.6g}" for lr in state["muon"]]))
        if "adam" in state:
            parts.append("adam=" + "/".join([f"{lr:.6g}" for lr in state["adam"]]))
        return " | ".join(parts)

    @torch.no_grad()
    def _initialize_memory_from_gt(self):
        """
        Fill memory with unique GT masks
        """
        memory_capacity_masks = int(self.cfg.memory_capacity)
        unique_threshold = float(self.cfg.memory_unique_threshold)

        if memory_capacity_masks <= 0:
            print("[Memory Init] skipped because memory_init_masks <= 0")
            return

        self.model.mask_memory.reset_memory()
        added_total = 0
        skipped_total = 0

        print(f"[Memory Init] inserting {memory_capacity_masks} GT masks " 
              f"before epoch 1 | unique_threshold={unique_threshold}")

        # More than one pass is allowed because duplicate masks may be skipped.
        max_passes = 2

        for pass_id in range(max_passes):
            if added_total >= memory_capacity_masks:
                break

            for image, gt, name in self.train_loader:
                gt_mask = gt[1].to(self.cfg.device)
                mask_names = name[1]

                remaining = memory_capacity_masks - added_total
                if remaining <= 0:
                    break

                B = gt_mask.shape[0]
                idx = torch.randperm(B, device=gt_mask.device)

                gt_mask = gt_mask[idx]
                mask_names = [mask_names[int(i)] for i in idx.detach().cpu().tolist()]

                added, skipped = self.model.mask_memory.add_unique_gt_batch(gt_masks=gt_mask, sample_ids=mask_names, max_add=remaining,
                                                                            unique_threshold=unique_threshold, min_fg_pixels=2)

                added_total += added
                skipped_total += skipped

                if added_total >= memory_capacity_masks:
                    break

        print(f"[Memory Init] done | "
              f"requested={memory_capacity_masks} | "
              f"inserted_unique={added_total} | "
              f"skipped_duplicate_or_invalid={skipped_total} | "
              f"memory_count={self.model.mask_memory.count}")

        if added_total < memory_capacity_masks:
            print("[Memory Init Warning] Could not fill the requested number of unique masks. "
                  "Increase memory_init_max_passes, lower memory_unique_threshold, "
                  "or reduce memory_init_masks.")

    def train(self):
        # ====================================== Training ===========================================#
        model_score = 0.
        t = time.time()
        self.lr = self._lr_string(self.optimizer)

        # Model Train
        if os.path.isfile(self.net_path):
            Train_results = open(self.result_path + self.report_name + '_Train_result.csv', 'a', encoding='utf-8', newline='')
            twr = csv.writer(Train_results)

            Valid_results = open(self.result_path + self.report_name + '_Valid_result.csv', 'a', encoding='utf-8', newline='')
            vwr = csv.writer(Valid_results)
        else:
            Train_results = open(self.result_path + self.report_name + '_Train_result.csv', 'a', encoding='utf-8', newline='')
            twr = csv.writer(Train_results)
            Valid_results = open(self.result_path + self.report_name + '_Valid_result.csv', 'a', encoding='utf-8', newline='')
            vwr = csv.writer(Valid_results)
            twr.writerow(['Train_model', 'Model_type', 'Dataset', 'LR', 'Epochs', 'Aug_prob'])
            twr.writerow([self.report_name, self.model_type, self.dataset, self.lr, self.epochs, self.aug_prob])
            vwr.writerow(['Train_model', 'Model_type', 'Dataset', 'LR', 'Epochs', 'Aug_prob'])
            vwr.writerow([self.report_name, self.model_type, self.dataset, self.lr, self.epochs, self.aug_prob])

            if self.learning_mode == 'Dist':
                twr.writerow(['Epoch', 'LRs', 'Loss', 'CS'])
                vwr.writerow(['Epoch', 'LRs', 'Loss', 'CS'])
            elif self.learning_mode in ['Seed', 'Seg']:
                twr.writerow(['Epoch', 'LRs', 'Loss', 'Acc', 'Rc', 'Pr', 'F1', 'OIS', 'IoU', 'AIU', 'Dc'])
                vwr.writerow(['Epoch', 'LRs', 'Loss', 'Acc', 'Rc', 'Pr', 'F1', 'OIS', 'IoU', 'AIU', 'Dc'])

        # Training
        results = []
        self.learner = Learner(self.cfg, self.model, self.teacher, self.optimizer, self.grad_scaler, self.loss_eq, self.steps)
        if self.learning_mode == "Seg" and hasattr(self.model, "mask_memory"):
            self._initialize_memory_from_gt()
        if self.learning_mode == 'Dist':
            results = [["Loss",[],[]], ["CS",[],[]]]
        elif self.learning_mode in ['Seed', 'Seg']:
            results = [["Loss",[],[]], ["Acc",[],[]], ["Rc",[],[]], ["Pr",[],[]], ["F1",[],[]], ["OIS",[],[]], ["IoU",[],[]], ["AIU",[],[]], ["Dc",[],[]]]

        for epoch in range(self.epochs):
            # Print the report info
            self.lr = self._lr_string(self.optimizer)
            print(f"\nEpoch [{epoch+1}/{self.epochs}], LR: [{self.lr}] \n[Training]")
            self.report.write(f"\nEpoch [{epoch+1}/{self.epochs}], LR: [{self.lr}] \n[Training]")

            memory_selected_batches = int(self.cfg.memory_selected_batches)
            num_train_batches = len(self.train_loader)
            memory_selected_batches = min(memory_selected_batches, num_train_batches)
            memory_selections_done = 0

            evaluator = Evaluation(self.cfg)
            with tqdm(total=len(self.train_loader.dataset)) as pbar:
                for i, (image, gt, name) in enumerate(self.train_loader):
                    enable_memory_match = False
                    memory_selected_index = None

                    if self.learning_mode == "Seg" and memory_selections_done < memory_selected_batches:
                        remaining_batches = num_train_batches - i
                        remaining_budget = memory_selected_batches - memory_selections_done

                        # Online random selection of batches.
                        choose_this_batch = random.random() < (remaining_budget / max(remaining_batches, 1))
                        if choose_this_batch:
                            batch_size_now = image[0].shape[0]
                            memory_selected_index = random.randint(0, batch_size_now - 1)
                            enable_memory_match = True
                            memory_selections_done += 1

                    r = self.learner.learn(image, gt, train="train", enable_memory_match=enable_memory_match, memory_selected_index=memory_selected_index,
                                           sample_ids=name[1], )

                    memory_info = r[3] if len(r) > 3 else None
                    if self.learning_mode == "Seg" and memory_info is not None and enable_memory_match:
                        print("[Memory] "
                              f"epoch={epoch + 1} | batch={i} | "
                              f"selected_index={memory_info['selected_index']} | "
                              f"mask_id={memory_info['selected_sample_id']} | "
                              f"filename_hits={memory_info['filename_hits']} | "
                              f"used_filename_memory={memory_info['used_filename_memory']} | "
                              f"matched={memory_info['matched']} | "
                              f"best_sim={memory_info['best_sim']:.4f} | "
                              f"memory_count={memory_info['memory_count']}")

                    # Get metrices results
                    metrics = evaluator.metrics(r[0], r[1], r[2])

                    if self.lr_sch_type == 'LinearWarmupCosineDecay':
                        self.lr_sch.step()
                        self.lr = self._lr_string(self.optimizer)

                    pbar.update(image[0].shape[0])
                    pbar.set_postfix(**{'batch loss': r[2].item(), 'lr': self.lr})
                    
            metavg = evaluator.metrics_avg(metrics)
            for i in range(len(results)):
                results[i][1].append(metavg[i])

            if self.learning_mode == 'Dist':
                print('\n[R] Loss: %.4f, CS: %.4f' % (metavg[0], metavg[1]))
                self.report.write('\n[R] Loss: %.4f, CS: %.4f' % (metavg[0], metavg[1]))
                twr.writerow([epoch+1, self.lr, metavg[0], metavg[1]])
            elif self.learning_mode in ['Seed', 'Seg']:
                print('\n[R] Loss: %.4f, Acc: %.4f, Rc: %.4f, Pr: %.4f, F1: %.4f, OIS: %.4f, IoU: %.4f, AIU: %.4f, Dc: %.4f' % (
                    metavg[0], metavg[1], metavg[2], metavg[3], metavg[4], metavg[5], metavg[6], metavg[7], metavg[8]))
                self.report.write('\n[R] Loss: %.4f, Acc: %.4f, Rc: %.4f, Pr: %.4f, F1: %.4f, OIS: %.4f, IoU: %.4f, AIU: %.4f, Dc: %.4f' % (
                    metavg[0], metavg[1], metavg[2], metavg[3], metavg[4], metavg[5], metavg[6], metavg[7], metavg[8]))
                twr.writerow(
                    [epoch+1, self.lr, metavg[0], metavg[1], metavg[2], metavg[3], metavg[4], metavg[5], metavg[6], metavg[7], metavg[8]])

            # Clear unoccupied GPU memory after each epoch
            torch.cuda.empty_cache()
            
            # ========================== Validation ====================================#
            print('\n[Validating]')
            self.report.write('\n[Validating]')
            
            evaluator = Evaluation(self.cfg)
            with tqdm(total=len(self.valid_loader.dataset)) as pbar:
                for i, (image, gt, name) in enumerate(self.valid_loader):
                    r = self.learner.learn(image, gt, train='valid')
                    
                    # Get metrices results
                    metrics = evaluator.metrics(r[0], r[1], r[2])
                    
                    pbar.update(image[0].shape[0])
                    pbar.set_postfix(**{'batch loss': r[2].item()})
                    
            metavg = evaluator.metrics_avg(metrics)
            for i in range(len(results)):
                results[i][2].append(metavg[i])

            if self.learning_mode == 'Dist':
                print('\n[R] Loss: %.4f, CS: %.4f' % (metavg[0], metavg[1]))
                self.report.write('\n[R] Loss: %.4f, CS: %.4f' % (metavg[0], metavg[1]))
                vwr.writerow([epoch+1, self.lr, metavg[0], metavg[1]])
            elif self.learning_mode in ['Seed', 'Seg']:
                print('\n[R] Loss: %.4f, Acc: %.4f, Rc: %.4f, Pr: %.4f, F1: %.4f, OIS: %.4f, IoU: %.4f, AIU: %.4f, Dc: %.4f' % (
                    metavg[0], metavg[1], metavg[2], metavg[3], metavg[4], metavg[5], metavg[6], metavg[7], metavg[8]))
                self.report.write('\n[R] Loss: %.4f, Acc: %.4f, Rc: %.4f, Pr: %.4f, F1: %.4f, OIS: %.4f, IoU: %.4f, AIU: %.4f, Dc: %.4f' % (
                    metavg[0], metavg[1], metavg[2], metavg[3], metavg[4], metavg[5], metavg[6], metavg[7], metavg[8]))
                vwr.writerow(
                    [epoch+1, self.lr, metavg[0], metavg[1], metavg[2], metavg[3], metavg[4], metavg[5], metavg[6], metavg[7], metavg[8]])

            # Decay learning rate
            if self.lr_sch_type == "ReduceLROnPlateau":
                self.lr_sch.step(metavg[1]) if self.learning_mode == 'Dist' else self.lr_sch.step(metavg[4])
            elif self.lr_sch_type == "CosineAnnealingWarmRestarts":
                self.lr_sch.step(epoch + 1)

            # Save Best Model
            if self.learning_mode == 'Dist' and metavg[1] > model_score:
                model_score = metavg[1]
                print('\nBest %s model score : %.4f' % (self.model_type, model_score))
                self.report.write('\nBest %s model score : %.4f' % (self.model_type, model_score))
                state_dict = self.model.state_dict()
                torch.save(state_dict, self.net_path)
            elif self.learning_mode in ['Seed', 'Seg'] and metavg[4] > model_score:
                model_score = metavg[4]
                print('\nBest %s model score : %.4f' % (self.model_type, model_score))
                self.report.write('\nBest %s model score : %.4f' % (self.model_type, model_score))
                state_dict = self.model.state_dict()
                torch.save(state_dict, self.net_path)
                
            # Clear unoccupied GPU memory after each epoch
            torch.cuda.empty_cache()
        displayfigures(results, self.result_path, self.report_name)
        
        Train_results.close()
        Valid_results.close()
        elapsed = time.time() - t
        print("\nElapsed time: %f seconds.\n\n" % elapsed)
        self.report.write("\nElapsed time: %f seconds.\n\n" % elapsed)
        self.report.close()

    def test(self):
        # ===================================== Test ====================================#
        # Load Trained Model
        if os.path.isfile(self.net_path):
            self.model.load_state_dict(torch.load(self.net_path, map_location='cpu', weights_only=False))
            self.model = self.model.to(self.cfg.device)
            print('%s is Successfully Loaded from %s' % (self.model_type, self.net_path))
            self.report.write('\n%s is Successfully Loaded from %s' % (self.model_type, self.net_path))
        else:
            print(self.net_path + " is not exist")
            print("Trained model NOT found, Please train a model first")
            self.report.write("\nTrained model NOT found, Please train a model first")
            return

        # Print the report info
        print('\n[Testing]')
        self.report.write('\n[Testing]')
        
        elapsed = 0.  # Time of inference
        Rc_curve = 0.
        Pr_curve = 0.
        Rc_all = []
        Pr_all = []
        results = []
        self.learner = Learner(self.cfg, self.model, self.teacher, self.optimizer, self.grad_scaler, self.loss_eq)
        if self.learning_mode == 'Dist':
            results = [["Loss",[]], ["CS",[]]]
        elif self.learning_mode in ['Seed', 'Seg']:
            results = [["Loss",[]], ["Acc",[]], ["Rc",[]], ["Pr",[]], ["F1",[]], ["OIS",[]], ["IoU",[]], ["AIU",[]], ["Dc",[]]]

        evaluator = Evaluation(self.cfg)
        with tqdm(total=len(self.test_loader.dataset)) as pbar:
            for i, (image, gt, name) in enumerate(self.test_loader):
                t = time.time() # Time of inference
                r = self.learner.learn(image, gt, train='test')
                elapsed = (time.time() - t)
                
                # Get metrices results
                metrics = evaluator.metrics(r[0], r[1], r[2])
                if self.learning_mode == 'Seg':
                    Rc_all.append(metrics[3])
                    Pr_all.append(metrics[5])
                
                pbar.update(image[0].shape[0])
                pbar.set_postfix(**{'batch loss': r[2].item()})
            
        metavg = evaluator.metrics_avg(metrics)
        for i in range(len(results)):
            results[i][1].append(metavg[i])
                
        elapsed = elapsed / (r[0].size(0))
        if self.learning_mode == 'Seg':
            Rc_curve, Pr_curve = PRC(Pr_all, Rc_all, self.result_path, self.report_name)
            PRC_report = open(self.result_path+self.report_name+'_PRC.txt','a+')
            PRC_report.write('\n\n Recall = '+str(list(Rc_curve)))
            PRC_report.write('\n Precision = '+str(list(Pr_curve)))
            PRC_report.close()

        self.lr = self._lr_string(self.optimizer)
        f = open(os.path.join(self.result_path, 'Test_result.csv'), 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
        if self.learning_mode == 'Dist':
            wr.writerow(
                ['Report_file', 'Model_type', 'Dataset', 'Loss', 'CS', 'Time of inference', 'LRs', 'Epochs', 'Aug_prob'])
            wr.writerow(
                [self.report_name, self.model_type, self.dataset, metavg[0], metavg[1], elapsed, self.lr, self.epochs, self.aug_prob])
        elif self.learning_mode in ['Seed', 'Seg']:
            wr.writerow(
                ['Report_file', 'Model_type', 'Dataset', 'Loss', 'Acc', 'Rc', 'Pr', 'F1', 'OIS', 'IoU', 'AIU', 'Dc',
                 'Time of inference', 'LRs', 'Epochs', 'Aug_prob'])
            wr.writerow([self.report_name, self.model_type, self.dataset, metavg[0], metavg[1], metavg[2], metavg[3], metavg[4], metavg[5],
                 metavg[6], metavg[7], metavg[8], elapsed, self.lr, self.epochs, self.aug_prob])
        f.close()

        print('Results have been Saved')
        self.report.write('\nResults have been Saved\n\n')

        self.report.close()
