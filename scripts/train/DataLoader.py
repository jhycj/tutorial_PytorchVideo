import os
import pytorch_lightning 
import pytorchvideo.data
import torch.utils.data

from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip

from pytorchvideo.transforms import (
  ApplyTransformToKey, 
  #Normalize, 
  RandomShortSideScale, 
  RemoveKey,
  ShortSideScale, 
  UniformTemporalSubsample 
)

class KineticsDataModule(pytorch_lightning.LightningDataModule): 

  # Dataset Configuration 
  _DATA_PATH = '../../dataset'
  _CLIP_DURATION = 2 
  _BATCH_SIZE = 8 
  _NUM_WORKERS = 4

  def train_data_loader(self): 
    """Create the Kinetics train partition from the list of video labels 
    in {self._DATA_PATH}/train
    """
    train_dataset = pytorchvideo.data.Kinetics(
      data_path = os.path.join(self._DATA_PATH,"train"), 
      clip_sampler = pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
      decode_audio = False,
      transform = get_transform() 
    )

    train_data_loader = torch.utils.data.DataLoader(
      train_dataset, 
      batch_size = self._BATCH_SIZE, 
      num_workers = self._NUM_WORKERS
    )

    return train_data_loader 

  def val_data_loader(self): 

    val_dataset = pytorchvideo.data.Kinetics(
      data_path = os.path.join(self._DATA_PATH, "valid"),
      clip_sampler = pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
      decode_audio = False
    )

    val_data_loader = torch.utils.data.DataLoader(
      val_dataset, 
      batch_size = self._BATCH_SIZE, 
      num_workers = self._NUM_WORKERS 
    )
    return val_data_loader 

def get_transform(): 
  
  transform = Compose(
    [
    ApplyTransformToKey(
      key="video",
      transform=Compose(
      [
        UniformTemporalSubsample(8),
        #Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        RandomShortSideScale(min_size=256, max_size=320),
        RandomCrop(244),
        RandomHorizontalFlip(p=0.5),
      ]
      ),
    ),
    ]
  )
  
  return transform 