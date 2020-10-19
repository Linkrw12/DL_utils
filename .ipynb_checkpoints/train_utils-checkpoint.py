import random
import torch
import numpy as np

class SAETrainer:
    """
    A wrapper class to hold all functions related to training SAEs
    """
    
    def __init__(self, learner, epochs):
        self.learner = learner
        self.epochs  = epochs
        
    def _find_steepest_descent():
        """
        
        """
        
        pass
    
    def find_best_lr(self):
        """
        
        """
        
        # x = self.learner.recorder.lrs
        # y = self.learner.recorder.losses
        # min_lr, max_lr = self._find_steepest_descent()
        
        pass
    
    def freeze_weights(self, step):
        """
        
        """
        self.learner.model.eval()
        
        self.learner.model.hidden_layers[step-1].layer.weight.requires_grad = False
        self.learner.model.hidden_layers[step-1].layer.bias.requires_grad = False
        
        self.learner.model.hidden_layers[-step].layer.weight.requires_grad = False
        self.learner.model.hidden_layers[-step].layer.bias.requires_grad = False
        
        self.learner.model.train()
    
    def train_sae(self):
        """

        """

        train_steps = range(1, self.learner.model.self._temp_layers+1)

        for step in train_steps:

            self.learner.model.train_step = step
            
            # Find new learning rate
            self.learner.lr_find()
            
            # Get new slice for min-max lr
            min_lr, max_lr = self.find_new_lr(self.learner)
            
            # Fit the model using your specified learner and epochs
            self.learner.fit_one_cycle(self.epochs, max_lr = slice(min_lr, max_lr))
            
            # Freeze weights
            self.freeze_weights(step)
            
            if step == self.learner.model.self._temp_layers:
                learner.model

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
    
def find_lr_range(learner):
    # x = learner.recorder.lrs
    # y = learner.recorder.val_losses
    # Find y_min
    # 
    pass