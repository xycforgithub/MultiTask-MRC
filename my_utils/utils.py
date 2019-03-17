import random
import torch
import numpy
import json

class AverageMeter(object):
    """Computes and stores the average and current value."""
    # adapted from: https://github.com/facebookresearch/DrQA
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def state_dict(self):
        return {
            'val':self.val,
            'sum':self.sum,
            'count':self.count,
            'avg':self.avg
        }
    def load_state_dict(self,sd):
        self.val = sd['val']
        self.sum = sd['sum']
        self.count = sd['count']
        self.avg = sd['avg']

def set_environment(seed, set_cuda=False):
    
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and set_cuda:
        torch.cuda.manual_seed_all(seed)

def repeat_save(func,*args,**kwargs):
    for i in range(10):
        try:
            func(*args,**kwargs)
            break
        except:
            print('save failed. trying again.')

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data