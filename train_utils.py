import random
import torch
import numpy as np

import fastai
from fastai.callback.all import LRFinder 

class SAETrainer:
    """
    A wrapper class to hold all functions related to training SAEs
    """
    
    def __init__(self, epochs, seed):
        self.epochs  = epochs
        self.seed    = seed
    
	def freeze_single_layer(self, layer):
		for param in layer.parameters():
			param.requires_grad = False
	
    def freeze_weights(self, learner, step):
        """
        
        """
        learner.model.eval()
        self.freeze_single_layer(learner.model.hidden_layers[step-1])
        self.freeze_single_layer(learner.model.hidden_layers[-step])
        learner.model.train()
    
    def fit_one_cycle(self, learner, lr_min, lr_steep):
        if lr_min >= lr_steep:
            learner.fit_one_cycle(self.epochs, max_lr = slice(lr_steep, lr_min))
        else:
            learner.fit_one_cycle(self.epochs, max_lr = slice(lr_min, lr_steep))
    
    def train_sae_step(self, learner, step):
        """
        
        """
        
        # Change behavior of training 
        learner.model.train_step = step
            
        lr_min, lr_steep = learner.lr_find()

        # Fit the model using your specified learner and epochs
        self.fit_one_cycle(learner, lr_min, lr_steep)
        self.freeze_weights(learner, step)
        
        
    def plot_train_val_loss(self, learner, title, savefile = None):
        
        fig, ax = plt.subplots()
        learner.recorder.plot_loss()
        
        sbn.despine()
        plt.ylabel("MSE Loss")
        plt.xlabel("Batches")
        plt.title(title)
        
        if savefile:
            plt.savefig(savefile, dpi = 300)
        
        plt.show()
        
    def train_sae(self, learner):
        
        set_seed(self.seed)
        
        STEPS = len(learner.model.hidden_layer_arg)
        train_steps = range(1, STEPS+1)
        
        for step in train_steps:
            self.train_sae_step(learner, step)
            self.plot_train_val_loss(learner,
                                     title = f"SAE Phase {step} Train-Val Loss")
            
        learner.model.train_step = 0
        learner.model.eval()

def set_seed(seed):
    """
    Sets all seed values to a consistant value for CPU and GPU applications. 
    
    Code found here:
    https://docs.fast.ai/dev/test.html#getting-reproducible-results
    """

    # python RNG
    random.seed(seed)

    # pytorch RNGs
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # numpy RNG
    np.random.seed(seed)
    
def freeze_sae_layers(learner, train_step):
    learner.model.eval()

    learner.model.hidden_layers[train_step-1].layer.weight.requires_grad = False
    learner.model.hidden_layers[train_step-1].layer.bias.requires_grad = False

    learner.model.hidden_layers[-train_step].layer.weight.requires_grad = False
    learner.model.hidden_layers[-train_step].layer.bias.requires_grad = False
    
    learner.model.train()