from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __getitem__(self, index):
        train_data = self.train[index]
        label_data = self.label[index].reshape(self.board_size*self.board_size)
        return train_data, label_data

    def __init__(self, train,label,board_size):
        self.train = train
        self.label = label
        self.board_size=board_size
        # self.name_r = os.listdir(os.path.join(path, 'real'))

    def __len__(self):
        return len(self.train)