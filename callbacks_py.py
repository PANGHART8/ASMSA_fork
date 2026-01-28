import torch
import numpy as np
import copy
import csv
import os
from callbacks_py_viz import Visualizer

class Callbacks_py:
    def __init__(self, model, log_dir, monitor="val_ae_loss", patience_stop=15, patience_lr=7, factor=0.5, freq=20, inputs=None):
        self.model = model
        self.monitor = monitor
        self.freq = freq
        self.inputs = inputs
        self.log_dir = log_dir
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.log_file = os.path.join(log_dir, 'training_log.csv')
        self.headers_written = False

        # Early Stopping, LR
        self.patience_stop = patience_stop
        self.best_loss = float('inf')
        self.counter_stop = 0
        self.best_model_weights = None
        
        self.patience_lr = patience_lr
        self.factor = factor
        self.counter_lr = 0
        
        self.visualizer = Visualizer(figsize=(12, 3))

    def on_epoch_end(self, epoch, logs):
        current_val_loss = logs.get(self.monitor)
        lr = self.model.optimizer.param_groups[0]['lr']
        min_delta = 1e-6

        all_data = {**logs, 'epoch': epoch, 'lr': lr}
        
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_data.keys())
            if not self.headers_written:
                writer.writeheader()
                self.headers_written = True
            writer.writerow(all_data)

        if epoch % self.freq == self.freq - 1:
            self.model.eval()
            with torch.no_grad():
                lows = self.model.call(self.inputs)
                self.visualizer.make_visualization(lows, title=f"Epoch {epoch}")
            self.model.train()

        if current_val_loss < (self.best_loss - min_delta):
            self.best_loss = current_val_loss
            self.counter_stop = 0
            self.counter_lr = 0
            self.best_model_weights = copy.deepcopy(self.model.state_dict())
        else:
            self.counter_stop += 1
            self.counter_lr += 1

            if self.counter_lr >= self.patience_lr:
                new_lr = lr * self.factor
                for param_group in self.model.optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"\n[LR] Zníženie na {new_lr}")
                self.counter_lr = 0

            if self.counter_stop >= self.patience_stop:
                if self.best_model_weights is not None:
                    self.model.load_state_dict(self.best_model_weights)
                print(f"\n[EarlyStopping] Koniec v epoche {epoch}. Obnovené najlepšie váhy.")
                return True 
        
        return False