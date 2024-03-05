import os
from PIL import Image
from torch.utils.data.dataset import Dataset


class TripletDataset(Dataset):
    """
    Each triplet has its folder with the corresponding 3 images.
    Structure of the triplet dataset:
        triplet_<number>
            -   1.png
            -   2.png
            -   3.png
    """
    def __init__(self, triplets_path, transform):
        self.triplets_path = triplets_path  # The path to the folder containing the triplets
        self.transform = transform  # This transforms to tensor and normalizes the images
        self.triplets = list(os.listdir(triplets_path)) # List of all the triplets in the folder


    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, index):
        current_path = self.triplets[index] # e.g. triplet triplet_000010
        
        # Make sure you understand the __init__ function and the structure of the data first.

        # TASK 1: Read the triplet and make sure you use the self.transform
        frame_1 = Image.open(os.path.join(self.triplets_path, current_path, '1.png'))   # Read the first frame
        frame_1 = self.transform(frame_1)  # Apply the transform (to tensor and normalize) to the first frame

        frame_2 = Image.open(os.path.join(self.triplets_path, current_path, '2.png'))
        frame_2 = self.transform(frame_2)

        frame_3 = Image.open(os.path.join(self.triplets_path, current_path, '3.png'))
        frame_3 = self.transform(frame_3)

        return frame_1, frame_2, frame_3
