class PwTFDataset(Dataset):

    def __init__(self,file_list):

        self.files=file_list

    def __len__(self):

        return len(self.files)

    def __getitem__(self,idx):

        path=self.files[idx]

        feature=np.load(path).astype(np.float32)

        label=int(

            os.path.basename(path)

            .split("_")[-1]

            .replace(".npy","")

        )

        feature=torch.tensor(feature)

        label=torch.tensor(
            label,
            dtype=torch.float32
        )

        return feature,label

#classify datasets into training and validation

train_dataset = PwTFDataset(train_files)

val_dataset = PwTFDataset(val_files)
