from collections import Counter

from cuml.cluster import DBSCAN
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

class _ClusterSplitter:
    """
    Creates the train-test-val splits from DBSCAN clusters to maximize model robustness. 
    
    inputs:
        clusters: a DBSCAN object that was applied to the encoded_sequence_tensor that is applied
        test: the proportion of samples assigned to the training set (default = 0.80)
        test: the proportion of samples assigned to the testing set (default = 0.10)
         val: the proportion of samples assigned to the validation set (default = 0.10)
    usage:
        >>> clusters = DBSCAN().fit(<encoded_sequence_tensor>)
        >>> assigner = ClusterSplitter(clusters)
        >>> train, test, val = assigner.train_test_val_split(<encoded_sequence_tensor>)
    
    returns:
        train: a nested list of length(k) holding training samples for each k-fold split. 
                Meant to be used as an input for Torch.Dataset
         test: a nested list of length(k) holding testing samples for each k-fold split
          val: a nested list of length(k) holding validation samples for each k-fold split
    """
    
    def __init__(self, clusters, k = 5, train = 0.80, 
                 test = 0.10, val = 0.10, seed = 1231):
        
        assert int(sum([train, test, val])) == 1, f"Split proportions need to sum up to 1.0. Got: train: {train}, test: {test}, val: {val}"
        assert type(clusters) == DBSCAN, f"Cluster nput needs to be a DBSCAN object. Input type is {type(clusters)}"
        assert type(k) == int, f"k-fold input needs to be an int. Input type is {type(k)}"
        assert type(seed) == int, f"seed input needs to be an int. Input type is {type(k)}"
        
        self.train = train
        self.test  = test
        self.val   = val
        self.k     = k
        self.seed  = seed
        
        # Clustering information
        self.cluster_labels       = clusters.labels_
        self.cluster_counts       = Counter(self.cluster_labels)
        self.class_range          = list(range( min(self.cluster_labels), max(self.cluster_labels)+1 ))
                
        # If sum of the train-test-val samples do not equal the sample size,
        # the _correct_sample_splits function will be run
        self.total_samples = len(self.cluster_labels)
        self.train_size    = round(self.total_samples * self.train)
        self.test_size     = round(self.total_samples * self.test)
        self.val_size      = round(self.total_samples * self.val)
        
        # Holds the indeces to assign samples to each cluster
        self.train_idx = list()
        self.test_idx  = list()
        self.val_idx   = list()
        
        # Holds the train-test-val splits for each k-fold
        self.folds = list()
        
    def _correct_sample_splits(self):
        """
        Calculates the difference between the sample splits and the actual
        number of samples. This difference is then used to modify the number
        of training samples to rectify the difference        
        """
        
        total_split_samples = sum([self.train_size, self.test_size, self.val_size])
        delta = total_split_samples - self.total_samples
        self.train_size -= delta
        
        assert total_split_samples == sum([self.train_size, self.test_size, self.val_size])
    
    def _assign_clusters_to_specific_split(self):
        """
        
        """     
        # Create the train-test-val splits for each index
        k_fold_seperator = KFold(n_splits = self.k, random_state = self.seed)
                
        folds = k_fold_seperator.split(self.class_range)
        folds = list(map(list, folds))
        
        # Replaces indeces with cluster classes
        for k in folds: 
            for split in k: 
                for i in range(len(split)): 
                    split[i] = self.class_range[split[i]]
        
        ##########################################
        # FIX 0.5 FOR NON 80-10-10 SPLIT SCHEMES #
        ##########################################
        folds = [[fold[0]] + train_test_split(fold[1], test_size = 0.5, random_state = self.seed) for fold in folds] # Adds the val splits by dividing
        
        self.folds = folds
                
        for k in range(self.k):
            self.train_idx.append( [ (idx, label) for idx, label in enumerate(self.cluster_labels) if label in self.folds[k][0] ] )
            self.test_idx.append( [ (idx, label) for idx, label in enumerate(self.cluster_labels) if label in self.folds[k][1] ] )
            self.val_idx.append( [ (idx, label) for idx, label in enumerate(self.cluster_labels) if label in self.folds[k][2] ] )
            
    def _rectify_split_size_differences(self):
        """
        Function designed to find all train-test-val splits that are 
        overfilled or underfilled with samples. It then corrects the
        differences by moving excess samples from overfilled splits 
        to underfilled splits to correct these differences. Any remaining
        sample excess gets moved to test split.
        
        It has four main steps:
            1. Find which train-test-val splits are overfilled or underfilled
               from their respective clusters

            2. Sort the train-test-val index lists by quantity, so clusters
               with the most samples get placed at the end of the list

            3. Move samples from overfilled splits to those with underfilled
               splits until prespecified split sample size is achieved in
               the underfilled

            4. All remaining overfill gets moved into the test split
        
        ------------------------------------------------------
        NOTE: When experimenting with subdividing large clusters
        into smaller clusters, DBSCAN would not split them into
        smaller clusters, even when the min cluster size was
        5 samples/cluster. This gave jutification for our
        Furhter subdivion method
        ------------------------------------------------------
        """
        
        def _calculate_final_split_current_split_difference(k):
            """
            Function to calculate the difference between the final
            train-test-val split sizes and the current train-test-val
            split sizes. 
            """
            split_delta = dict()
            
            data = ((self.train_idx, self.train_size, 'train'),
                    (self.test_idx, self.test_size, 'test'),
                    (self.val_idx, self.val_size, 'val'))
            
            for split, desired_size, key in data:
                
                split_size = len(split[k])
                
                delta = split_size - desired_size
                
                split_delta[key] = delta
                
            return split_delta
        
        def _reverse_sort_list_by_quantity(l):
            """
            Function to sort a list by the quantity of objects within it.
            
            Returns input list where fewer objects get placed at the 
            beggining of the list and greater lists get placed at the
            end of the list. 
            
            l: a list object to sort
            
            ex) 
            >>> x = [(10, 'a'), (26, 'a'), (31, 'c'), (43, 'b'), (51, 'b'), (64, 'b')]
            >>> _reverse_sort_list_by_quantity(x)
                
            out:[(31, 'c'), (26, 'a'), (10, 'a'), (64, 'b'), (51, 'b'), (43, 'b')]
            """
            
            out = list()
            labels = [label for _,label in l]
            
            ordered_counts = Counter(labels).most_common()
            
            # Sorts by most common items
            for i in range(len(ordered_counts)):
                out.extend( (idx, label) for idx, label in l if label == ordered_counts[i][0] )
                
            out.reverse()
            
            return out 
        
        def _move_samples_to_another_list(l1, l2, delta):
            """
            Function is responsible for moving items from one split to another split. There is
            no output from the function, but instead modifies the variables holding each list
            to 
            
            l1: A list object that moves the last <delta> objects to the other l2
            l2: A list object that recieves the last <delta> objects from l1
            delta: An int specifying how many items to move from l1 to l2
            
            ex)
            >>> x = ['a','b','c','d']
            >>> y = [1,2,3,4,5]
            >>> _move_samples_to_another_list(x, y, delta = 2)
            >>> x
            ['a','b']
            >>> y
            [1,2,3,4,5,'c','d']
            """
            assert type(l1) == list and type(l2) == list, "The items need to be a list object"
            assert type(delta) == int, f"delta can only be an int. Function got (delta = {delta} ; {type(delta)})"
            assert delta < len(l1), f"the number of moved items needs to be less than the length of the list"
            assert delta > 0, "the numbers of moved items must be greater than 0"
            
            moving_items = l1[-delta:]
            l2.extend(moving_items)
            del l1[-delta:]
        
        def _find_split_w_most_overflow(d):
            return max(d, key=d.get)
        
        def _find_split_w_most_underflow(d):
            return min(d, key=d.get)
        
        for k in range(self.k):
            
            # 1. Find split over/underflow
            split_delta = _calculate_final_split_current_split_difference(k)
            
            # 2. Sort clusters by overflowed splits
            self.train_idx[k] = _reverse_sort_list_by_quantity(self.train_idx[k])
            self.test_idx[k]  = _reverse_sort_list_by_quantity(self.test_idx[k])
            self.val_idx[k]   = _reverse_sort_list_by_quantity(self.val_idx[k]) 
            
            # 3. Move overfill into underfilled splits
            proper_fill_count = list(split_delta.values()).count(0)
            
            while proper_fill_count < 2:
                
                overfill_split  = _find_split_w_most_overflow(split_delta)
                underfill_split = _find_split_w_most_underflow(split_delta)
                
                # Calculate how many samples to move from overfilled split to underfilled split
                # Move until underfill or overfill reaches its full size
                split_diff          = split_delta[overfill_split] - split_delta[underfill_split]
                underfill_zero_diff = eval(f"self.{underfill_split}_size - len(self.{underfill_split}_idx[k])")
                delta               = split_diff - underfill_zero_diff
                
                # Moves samples from the overfilled split to underfilled split
                eval(f"_move_samples_to_another_list(self.{overfill_split}_idx[k], self.{underfill_split}_idx[k], delta)")
                
                # Reset the split dictionary differences after sample movement
                split_delta       = _calculate_final_split_current_split_difference(k)
                proper_fill_count = list(split_delta.values()).count(0)
                
            # 4. Move remaining overfill into test split
            # If no remaining overflow, this doesn't do anything
            overfill_split = _find_split_w_most_overflow(split_delta)
            
            # If the overfill if already test or the overfill is 0, then ignore this
            # 0 ends up breaking the code
            if overfill_split != 'test' and split_delta[overfill_split] > 0:
                 eval(f"_move_samples_to_another_list(self.{overfill_split}_idx[k], self.test_idx[k], {split_delta[overfill_split]})")
                    
    def _put_seqs_into_splits(self, cluster_input):
        """
        
        ex)
        >>> x = ['a','b','c','d']
        >>> self.train_idx = [0,2]
        >>> self.test_idx  = [1]
        >>> self.val_idx   = [3]
        >>> self.k = 1
        >>> train, test, val = _put_seqs_into_splits(x)
        >>> print(f"train: {train}, test: {test}, val: {val}")
        train: ['a','c'], test: ['b'], val: ['d']
        """
        
        assert self.k == len(self.train_idx), "the k-fold number does not match the number of splits in the train-test-val sets"
        
        train = list()
        test = list()
        val = list()
        
        for k in range(self.k):
            
            print(f"SPLIT {k} TRAIN DIST:")
            print(f"\t{Counter([label for _,label in self.train_idx[k]])}")
            print(f"SPLIT {k} TEST DIST:")
            print(f"\t{Counter([label for _,label in self.test_idx[k]])}")
            print(f"SPLIT {k} VAL DIST:")
            print(f"\t{Counter([label for _,label in self.val_idx[k]])}")
            print('-'*50)
            
            train.append([cluster_input[i] for i,_ in self.train_idx[k]])
            test.append([cluster_input[i] for i,_ in self.test_idx[k]])
            val.append([cluster_input[i] for i,_ in self.val_idx[k]])
        
        return train, test, val
    
    def train_test_val_split(self, cluster_input):
        """
        The main class function that creates 5-fold splits and assigns each
        sequence to their corresponding split. Also rectifies any differences
        between the 
        """
        
        # Calculate how many samples are needed for each train-test-val group
        self._correct_sample_splits()
        
        # use random to assign the number of clusters for each group
        self._assign_clusters_to_specific_split()
        
        # Fix the size differences between the current and desired train-test-val splits
        self._rectify_split_size_differences()
        
        # Split into its actual outputs
        train, test, val = self._put_seqs_into_splits(cluster_input)

        return train, test, val
    
class Train_test_val_splitter():
    """
    
    """
    
    def __init__(self, train_size, test_size, val_size, k = 5, 
                 dbscan_eps = 3, dbscan_min_cluster_samples = 50, seed = 1231):
        
        assert (train_size + test_size + val_size) == 1.0
        
        # 
        self.train_size = train_size
        self.test_size  = test_size
        self.val_size   = val_size
        
        # DBSCAN parameters
        self.dbscan_eps                 = dbscan_eps
        self.dbscan_min_cluster_samples = dbscan_min_cluster_samples
        
        # Misc. parameters. Random state seed and how many k-folds (5 default)
        self.seed = seed
        self.k = k
        
        
    def __call__(self, encoded_input):
        
        clusters = DBSCAN(eps=self.dbscan_eps, 
                          min_samples=self.dbscan_min_cluster_samples).fit(encoded_input)
        
        assigner = _ClusterSplitter(clusters, k = self.k, train = self.train_size, 
                                    test = self.test_size, val = self.val_size, 
                                    seed = self.seed)
        
        train, test, val = assigner.train_test_val_split(encoded_input)
        
        return train, test, val