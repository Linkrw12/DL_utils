from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from itertools import product

class _ActivationDropoutBase(nn.Module):
    """
    These arguments are everywhereeeeeee, just inheriting these
    so I don't have to keep typing them
    """
    
    def __init__(self, dropout, activation):
        super().__init__()
        
        self.activation = activation
        self.dropout    = dropout

class DenseWrapper(_ActivationDropoutBase):
    """
    TO DO:
        > Incorperate batch norm
        
    Used for easy parameter optimization when creating layers and a cleaner
    overall look
    """
    
    def __init__(self, in_features, out_features, 
                 activation = F.leaky_relu, 
                 batch_norm = True,
                 dropout = 0.2):
        super().__init__(dropout, activation)
        
        self.in_features  = in_features
        self.out_features = out_features
        
        self.include_batch_norm = batch_norm
        
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else (lambda x: x)
        
        self.layer = nn.Linear(self.in_features, self.out_features)
    
    def __repr__(self):
        return f"""DenseWrapper(in_features = {self.in_features}, \
out_features = {self.out_features}, \
activation = {self.activation.__name__}, \
batch norm = {self.include_batch_norm}, \
dropout = {self.dropout}, \
bias = {self.layer.bias is not None})"""
    
    def forward(self, x):
        
        return F.dropout(self.batch_norm(self.activation(self.layer(x))), 
                         p = self.dropout)
    
class Conv1dWrapper(_ActivationDropoutBase):
    """
    Used for easy parameter optimization when creating layers and a cleaner
    overall look
    """
    
    def __init__(self, input_shape, output_shape, filter_length,
                 activation = F.leaky_relu, 
                 dropout = 0.2):
        super().__init__(dropout, activation)
        
        self.input_shape  = input_shape
        self.output_shape = output_shape
        self.filter_width = filter_length
        
        self.layer = nn.Conv1d(self.input_shape, self.output_shape,
                               kernel_size = self.filter_width)
    
    def __repr__(self):
        return f"Conv1dWrapper(in_features = {self.input_shape}, out_features = {self.output_shape}, filter_width = {self.filter_width}, activation = {self.activation.__name__}, dropout = {self.dropout}, bias = {self.layer.bias is not None})"
    
    def forward(self, x):
        
        return F.dropout(self.activation(self.layer(x)), 
                         p = self.dropout)

class AutoEncoder(_ActivationDropoutBase):
    """
    ...
    """
    
    def __init__(self, input_output_shape,
                 hidden_layers = [128, 64, 32],
                 activation = F.leaky_relu, 
                 dropout = 0.2,
                 train = False):
        super().__init__(dropout, activation)
        
        assert len(hidden_layers) > 0, "The hidden layer list is empty. It needs at least 1 hidden layer to be an AutoEncoder"
        
        # Calculates the list of all layer sizes in the AE
        self.shape = input_output_shape
        self.layer_sizes = [self.shape] + hidden_layers + hidden_layers[::-1][1:] + [self.shape]
        
        # Create and store all layers
        self.hidden_layers = nn.ModuleList([DenseWrapper(self.layer_sizes[i-1], self.layer_sizes[i], 
                                                         self.activation, self.dropout)
                              for i in range(1,len(self.layer_sizes)) ])
        
        # By design, traditional SAEs have an odd number of layers Because of this, the below will always return the middle
        # most layer in the SAE. Used for returning the stopping point of an untrained SAE
        self._latent_layer = int(len(self.hidden_layers) / 2)
        self.latent_size   = hidden_layers[-1]
        
    def forward(self, x):
        
        if self.train:
            for layer in self.hidden_layers:
                x = layer(x)
        else:
            for layer in self.hidden_layers[:self._latent_layer]:
                x = layer(x)
                
        return x
    
class SAE(AutoEncoder):
    """
    Class using the improved CTF as input for the
    RNA and Protein sequence for both ImprovedCTF and ImprovedStructCTF
    """
    
    def __init__(self, input_output_shape : int,
                 hidden_layers = [128, 64, 32],
                 activation = F.leaky_relu, 
                 dropout    = 0.2,
                 train_step = 0):
        super().__init__(input_output_shape, hidden_layers, activation, dropout)
        
        # Parameter used for training. This must be less than the 
        # length of the hidden_layers input. This designates which 
        # section of the SAE to train. If train_step is 0, then
        # the SAE processes the input until the latent layer
        self.train_step = train_step
        
        # Temporary layers to assist with training the model. 
        self._temp_layers = nn.ModuleList([DenseWrapper(size, size, self.activation, self.dropout) 
                                           for size in hidden_layers[:-1]])

    def forward(self, x):
        
        if 0 < self.train_step <= len(self._temp_layers):
            layers = nn.ModuleList([*self.hidden_layers[:self.train_step], 
                                    self._temp_layers[self.train_step-1], 
                                    *self.hidden_layers[-self.train_step:]])
            
        elif self.train_step > len(self._temp_layers):
            layers = self.hidden_layers
            
        elif not self.train_step:
            layers = self.hidden_layers[:self._latent_layer]
            
        for layer in layers:
            x = layer(x)
            
        return x

class SAEBlock(_ActivationDropoutBase):
    """
    Combines the autoencoders from the DNA and RNA outputs into a
    block either incorperating the ImprovedCTF or 
    """
    
    def __init__(self, rna_sae, prot_sae,
                 layers  = [128, 64, 2],
                 hidden1 = 128,
                 hidden2 = 64,
                 dropout = 0.2,
                 activation = F.leaky_relu):
        super().__init__(dropout, activation)
        
        # inputted SAEs
        self.rna_sae  = rna_sae
        self.prot_sae = prot_sae
        
        self.layer_sizes = hidden
        self.layers      = nn.ModuleList([DenseWrapper(self.layer_sizes[i-1], self.layer_sizes[i], self.activation, self.dropout)
                                          for i in range(1,len(self.layer_sizes))])
        
    def forward(self, rna, protein):
        
        out1 = self.rna_sae(rna)
        out2 = self.prot_sae(protein)
        
        combined_output = torch.cat((out1, out2), 1)
        
        for layer in self.layers:
            output = layer(output)
            
        hidden = self.hidden(combined_output)
        output = self.output(hidden)
        
        return F.softmax(output, dim = 1)
    
class ConvBlock(_ActivationDropoutBase):
    """
    Used for easy parameter optimization when creating layers
    """
    
    def __init__(self, input_shape, 
                 filter_width1       = 6, 
                 filter_width2       = 5, 
                 hidden_channels1    = 45, 
                 hidden_channels2    = 64,
                 dropout             = 0.1,
                 maxpool_width       = 2,
                 activation          = F.leaky_relu,
                 convblock_out_nodes = 64):
        
        super().__init__(dropout, activation)
        
        # Model hyperparams
        self.input_shape         = input_shape
        self.maxpool_width       = maxpool_width
        self.hidden_channels1    = hidden_channels1
        self.hidden_channels2    = hidden_channels2
        self.filter_width1       = filter_width1
        self.filter_width2       = filter_width2
        self.convblock_out_nodes = convblock_out_nodes
        
        # Convolutional block layers
        self.conv1    = nn.Conv1d(1, self.hidden_channels1, kernel_size = self.filter_width1)
        self.maxpool1 = nn.MaxPool1d(self.maxpool_width)
        
        self.conv2    = nn.Conv1d(self.hidden_channels1, self.hidden_channels2, kernel_size = self.filter_width2)
        self.maxpool2 = nn.MaxPool1d(self.maxpool_width)
        
        self.conv3    = nn.Conv1d(self.hidden_channels2, self.hidden_channels1, kernel_size = self.filter_width1)
        self.output   = nn.Linear(self.hidden_channels1, self.convblock_out_nodes)
    
    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = F.batch_norm( self.maxpool1(out) )
        
        out = self.activation(self.conv2(out))
        out = F.batch_norm( self.maxpool2(out) )
                
        out = F.batch_norm( self.activation(self.conv3(out)) )
        
        out = torch.flatten(out, start_dim = 1)
                
        out = self.output(out)
        
        return out
    
class ConjointCNNModule(_ActivationDropoutBase):
    """
    Class used for 
    """
    
    def __init__(self, prot_input_shape, rna_input_shape,
                 hidden_channels1    = 45,
                 hidden_channels2    = 65,
                 filter_width1       = 6,
                 filter_width2       = 5,
                 maxpool_width       = 2,
                 convblock_out_nodes = 2,
                 hidden_dense        = 64,
                 dropout             = 0.2,
                 activation          = F.leaky_relu):
                 
        super().__init__(dropout, activation)
        
        # Conv parameters
        self.prot_shape       = prot_input_shape
        self.rna_shape        = rna_input_shape
        self.hidden_channels1 = hidden_channels1
        self.hidden_channels2 = hidden_channels2
        self.filter_width1    = filter_width1
        self.filter_width2    = filter_width2
        self.maxpool_width    = maxpool_width
        
        # Dense parameters for conv block outputs and 
        # ConjointCNN hidden layers
        self.convblock_out_nodes = convblock_out_nodes
        self.hidden_dense        = hidden_dense
        
        # Add cuda back in when done testing
        self.ProtConvBlock = ConvBlock(self.prot_shape, self.filter_width1, self.filter_width2, 
                                       self.hidden_channels1, self.hidden_channels2, 
                                       self.dropout, self.maxpool_width, self.activation,
                                       self.convblock_out_nodes)
        
        self.RNAConvBlock  = ConvBlock(self.rna_shape, self.filter_width1, self.filter_width2, 
                                       self.hidden_channels1, self.hidden_channels2, 
                                       self.dropout, self.maxpool_width, self.activation,
                                       self.convblock_out_nodes)
        
        self.dense  = DenseWrapper(self.convblock_out_nodes * 2, self.hidden_dense, self.activation, self.dropout)
        self.output = DenseWrapper(self.hidden_dense, 2, self.activation, self.dropout)

    def forward(self, protein, rna):
        
        prot_out = self.ProtConvBlock(protein)
        rna_out  = self.RNAConvBlock(rna)
        
        output  = torch.cat((prot_out, rna_out), 1)
        
        output  = self.dense(output)
        output  = self.output(output)
        
        return F.softmax(output, dim = 1)
    
class DenseBlock(_ActivationDropoutBase):
    """
    Dense block used at the end of RPITER
    """
    
    def __init__(self, dropout = 0.2,
                 hidden_one    = 16,
                 hidden_two    = 8,
                 activation    = F.leaky_relu):
        super().__init__(dropout, activation)
        
        # Model Params
        self.hidden_one = hidden_one
        self.hidden_two = hidden_two
        
        # Layer sizes
        self.input  = DenseWrapper(8, self.hidden_one, self.activation, self.dropout)
        self.hidden = DenseWrapper(self.hidden_one, self.hidden_two, self.activation, self.dropout)
        self.output = DenseWrapper(self.hidden_two, 2, self.activation, self.dropout)
        
        self.batch_norm1 = nn.BatchNorm1d(self.hidden_one)
        self.batch_norm2 = nn.BatchNorm1d(self.hidden_two)
        
    def forward(self, x):
        
        out = self.input(x)
        out = self.batch_norm1(out)
        
        out = self.hidden(out)
        out = self.batch_norm2(out)
        
        out = self.output(out)
        
        return F.softmax(out, dim = 1)
        
class RPITER(_ActivationDropoutBase):
    """
    These blocks by defualt have a 
    """
    
    def __init__(self, rna_sae, prot_sae, rna_struct_sae, prot_struct_sae, 
                 rna_input_shape         = 340, 
                 prot_input_shape        = 399,
                 rna_struct_input_shape  = 370,
                 prot_struct_input_shape = 438,
                 dense_block_hidden1     = 16,
                 dense_block_hidden2     = 8,
                 dropout                 = 0.2,
                 activation              = F.leaky_relu):
        super().__init__(dropout, activation)
        
        ##############################################
        # SAE inputs
        ##############################################
        self.rna_sae         = rna_sae
        self.prot_sae        = prot_sae
        self.rna_struct_sae  = rna_struct_sae
        self.prot_struct_sae = prot_struct_sae
        
        ##############################################
        # Input shapes for ConjointCNNs
        ##############################################
        self.rna_input_shape         = rna_input_shape 
        self.prot_input_shape        = prot_input_shape
        self.rna_struct_input_shape  = rna_struct_input_shape
        self.prot_struct_input_shape = prot_struct_input_shape
        
        self.hidden_channels1    = 45
        self.hidden_channels2    = 65
        self.filter_width1       = 6
        self.filter_width2       = 5
        self.maxpool_width       = 2
        self.convblock_out_nodes = 64
        self.hidden_dense        = 64
        
        ##########################################
        # Dense block params
        ##########################################
        self.dense_block_hidden1 = dense_block_hidden1
        self.dense_block_hidden2 = dense_block_hidden2
        
        ##########################################
        # SAEBlock params
        ##########################################
        self.hidden1 = 128
        self.hidden2 = 64
        
        #########################################################

        ##########################################
        # Sequence blocks
        ##########################################
        self.conjoint_cnn = ConjointCNNModule(prot_input_shape, rna_input_shape, 
                                              hidden_channels1    = self.hidden_channels1,
                                              hidden_channels2    = self.hidden_channels2,
                                              filter_width1       = self.filter_width1,
                                              filter_width2       = self.filter_width2,
                                              maxpool_width       = self.maxpool_width,
                                              convblock_out_nodes = self.convblock_out_nodes,
                                              hidden_dense        = self.hidden_dense,
                                              dropout             = self.dropout,
                                              activation          = self.activation)
        
        self.conjoint_sae = SAEBlock(rna_sae, prot_sae,
                                     hidden1 = self.hidden1,
                                     hidden2 = self.hidden2,
                                     dropout = self.dropout,
                                     activation = self.activation)
        
        ##########################################
        # Struct blocks
        ##########################################
        self.conjoint_struct_cnn = ConjointCNNModule(prot_struct_input_shape, rna_struct_input_shape, 
                                                     hidden_channels1    = self.hidden_channels1,
                                                     hidden_channels2    = self.hidden_channels2,
                                                     filter_width1       = self.filter_width1,
                                                     filter_width2       = self.filter_width2,
                                                     maxpool_width       = self.maxpool_width,
                                                     convblock_out_nodes = self.convblock_out_nodes,
                                                     hidden_dense        = self.hidden_dense,
                                                     dropout             = self.dropout,
                                                     activation          = self.activation)
        
        self.conjoint_struct_sae = SAEBlock(rna_struct_sae, prot_struct_sae,
                                            hidden1 = self.hidden1,
                                            hidden2 = self.hidden2,
                                            dropout = self.dropout,
                                            activation = self.activation)
        
        ##########################################
        # Final Dense Block
        ##########################################
        self.dense_block = DenseBlock(dropout    = self.dropout,
                                      hidden_one = self.dense_block_hidden1,
                                      hidden_two = self.dense_block_hidden2,
                                      activation = self.activation)
        
    def forward(self, combined_inputs):
        """
        
        """
        
        # For some reason, passes "True" into the input
        # print(combined_inputs)
        
        # Unpack inputs into their corresponding variables
        rna, prot, rna_struct, prot_struct = combined_inputs
        
        seq_cnn_out = self.conjoint_cnn(rna, prot)
        seq_sae_out = self.conjoint_sae(rna, prot)
        
        struct_cnn_out = self.conjoint_struct_cnn(rna_struct, prot_struct)
        struct_sae_out = self.conjoint_struct_sae(rna_struct, prot_struct)
        
        joint_results = torch.cat((seq_cnn_out, seq_sae_out, struct_cnn_out, struct_sae_out))
        
        out = self.dense_block(joint_results)
        
        return out