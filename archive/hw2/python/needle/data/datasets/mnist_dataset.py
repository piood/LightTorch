from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms
        with gzip.open(image_filename, 'rb') as img_f: 
            magic_num, num_images, rows, cols = np.frombuffer(img_f.read(16), dtype=np.uint32) # Read the header, read 16 bytes, convert to 4 32-bit unsigned integers
            num_images, rows, cols = num_images.byteswap(), rows.byteswap(), cols.byteswap()  # Reverse byte order, convert to little endian (x86)
            images = np.frombuffer(img_f.read(), dtype=np.uint8).reshape(num_images, rows * cols) # Read the remaining data, convert to image matrix
            images = images.astype(np.float32) / 255.0   # Normalize to 0-1
            self.images = np.vstack(images)
            
            self.rows = rows
            self.cols = cols

        # Read labels
        with gzip.open(label_filename, 'rb') as lbl_f:
            magic_num, num_labels = np.frombuffer(lbl_f.read(8), dtype=np.uint32)
            num_labels = num_labels.byteswap()  
            labels = np.frombuffer(lbl_f.read(), dtype=np.uint8)
            self.labels = labels
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        imgs = self.images[index]
        labels = self.labels[index]
        if len(imgs.shape) > 1:
            imgs = np.vstack([self.apply_transforms(img.reshape(self.rows, self.cols, 1)).reshape(imgs[0].shape) for img in imgs])
        else:
            imgs = self.apply_transforms(imgs.reshape(self.rows, self.cols, 1)).reshape(imgs.shape)
        return (imgs, labels)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION