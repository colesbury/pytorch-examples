import torch
import os
import PIL
from torch.utils.data import Dataset, DataLoader

"""
An example Dataset implementation and background data loading
"""

class MyImageDataset(Dataset):
  def __init__(self, path):
    self.filenames = os.listdir(path)

  def __getitem__(self, i):
    pic = Image.open(self.filenames[i])
    # convert to RGB tensor
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    img = img.view(pic.size[1], pic.size[0], len(pic.mode))
    img = img.transpose(0, 1).transpose(0, 2).float() / 255
    return img

  def __len__(self):
    return len(self.filenames)

dataset = MyImageDataset('./images')

# load the images one at a time
for img in dataset:
  print(img.size())  # e.g. torch.Size(3, 28, 28)

# load, batch, and shuffle images using multiple workers
loader = DataLoader(dataset, num_workers=4, batch_size=64, shuffle=True)
for batch in loader:
  print(batch.size())  #e.g. torch.Size(64, 3, 28, 28)
