import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, TensorDataset
from util import dataLoading, dataLoading_mat
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class KDDCupData(Dataset):
    def __init__(self, data_dir, mode):
        """Loading the data for train and test."""

        features, labels = dataLoading(data_dir, None)
        #In this case, "atack" has been treated as normal data as is mentioned in the paper
        normal_data = features[labels==0] 
        normal_labels = labels[labels==0]

        # n_train = int(normal_data.shape[0]*0.7)
        # ixs = np.arange(normal_data.shape[0])
        # np.random.shuffle(ixs)
        # normal_data_test = normal_data[ixs[n_train:]]
        # normal_labels_test = normal_labels[ixs[n_train:]]
        normal_data_train, normal_data_test, normal_labels_train, normal_labels_test = \
            train_test_split(normal_data, normal_labels, test_size = 0.3, random_state=0)
        
        if mode == 'train':
            # self.x = normal_data[ixs[:n_train]]
            # self.y = normal_labels[ixs[:n_train]]
            self.x = normal_data_train
            self.y = normal_labels_train
            
        elif mode == 'test':
            anomalous_data = features[labels==1]
            anomalous_labels = labels[labels==1]
            self.x = np.concatenate((anomalous_data, normal_data_test), axis=0)
            self.y = np.concatenate((anomalous_labels, normal_labels_test), axis=0)
        
        # self.x = normal_data_train
        # self.y = normal_labels_train
        # print(self.x.shape, self.y.shape)
        # self.train = Data.Dataset(self.x, self.y)

        # anomalous_data = features[labels==1]
        # anomalous_labels = labels[labels==1]
        # self.x_test = np.concatenate((anomalous_data, normal_data_test), axis=0)
        # self.y_test = np.concatenate((anomalous_labels, normal_labels_test), axis=0)
        # self.test = Data.Dataset(self.x_test, self.y_test)

    def __len__(self):
        """Number of images in the object dataset."""
        return self.x.shape[0]

    def __getitem__(self, index):
        """Return a sample from the dataset."""
        return np.float32(self.x[index]), np.float32(self.y[index])



def get_KDDCup99(args):
    """Returning train and test dataloaders."""
    data_dir = args.data_dir
    # data = KDDCupData(data_dir, 'train')
    # dataloader_train = DataLoader(data, batch_size=args.batch_size, 
    #                           shuffle=True, num_workers=0)
    
    # test = KDDCupData(data_dir, 'test')
    # dataloader_test = DataLoader(data, batch_size=args.batch_size, 
    #                           shuffle=False, num_workers=0)
    
    features, labels = dataLoading(data_dir, None)
    features = preprocessing.scale(features)
    # features, labels = dataLoading_mat(data_dir, None)
    print(features.shape,labels.shape)
    #In this case, "atack" has been treated as normal data as is mentioned in the paper
    normal_data = features[labels==0] 
    normal_labels = labels[labels==0]

    # n_train = int(normal_data.shape[0]*0.7)
    # ixs = np.arange(normal_data.shape[0])
    # np.random.shuffle(ixs)
    # normal_data_test = normal_data[ixs[n_train:]]
    # normal_labels_test = normal_labels[ixs[n_train:]]
    normal_data_train, normal_data_test, normal_labels_train, normal_labels_test = \
        train_test_split(normal_data, normal_labels, test_size = 0.3, random_state=0, stratify = normal_labels)
    anomalous_data = features[labels==1]
    anomalous_labels = labels[labels==1]
    normal_data_test = np.concatenate((anomalous_data, normal_data_test), axis=0)
    normal_labels_test = np.concatenate((anomalous_labels, normal_labels_test), axis=0)

    test_num = normal_labels_test.shape[0]
    
    normal_data_train, normal_labels_train = torch.from_numpy(np.array(normal_data_train)), torch.from_numpy(np.array(normal_labels_train))
    normal_data_test, normal_labels_test = torch.from_numpy(np.array(normal_data_test)), torch.from_numpy(np.array(normal_labels_test))
    data_train = TensorDataset(normal_data_train, normal_labels_train)
    data_test = TensorDataset(normal_data_test, normal_labels_test)

    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)
    
    
    dataloader_test = DataLoader(data_test, batch_size=test_num, 
                              shuffle=False, num_workers=0)
    return dataloader_train, dataloader_test