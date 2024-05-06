import pandas as pd
import csv
import torch

class Saver:
    def __init__(self, folder, fieldnames):
        self.fname = folder + '/metrics.csv'
        self.file = open(self.fname, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames)
        self.writer.writeheader()

    def save(self, metrics):
        for metric in metrics:
            if torch.is_tensor(metrics[metric]):
                metrics[metric] = metrics[metric].item()

        self.writer.writerow(metrics)
        if metrics['i'] % 1000 == 0:
            self.file.flush()


    
