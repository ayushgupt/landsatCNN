import torch
from torchvision import datasets

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

# EXAMPLE USAGE:
# instantiate the dataset and dataloader
# data_dir = "your/data_dir/here"
# dataset = ImageFolderWithPaths(data_dir) # our custom dataset
# dataloader = torch.utils.DataLoader(dataset)

# iterate over data
# for inputs, labels, paths in dataloader:
    # use the above variables freely
#     print(inputs, labels, paths)