import torch
import pytorch_lightning as pl  
import pytorchvideo.models as models 
import pytorchvideo.layers as layers 
from pytorchvideo.data import Kinetics
import os
import pytorchvideo.data
import torch.utils.data
from create_model import make_kinetics_resnet, make_slowfast
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, Resize, ToTensor
import torch.optim as optim 
import torch.nn as nn 
import torch.nn.functional as F 

from pytorchvideo.transforms import (
  ApplyTransformToKey, 
  #Normalize, 
  RandomShortSideScale, 
  RemoveKey,
  ShortSideScale, 
  UniformTemporalSubsample 
)

def get_transform(): 
  
  transform = Compose(
    [
    ApplyTransformToKey(
      key="video",
      transform=Compose(
      [
        UniformTemporalSubsample(8),
        Resize((244, 244)),
        #Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        #RandomShortSideScale(min_size=256, max_size=320),
        #RandomCrop(244),
        #RandomHorizontalFlip(p=0.5),
      ]
      ),
    ),
    ]
  )
  
  return transform 

class KineticsDataModule(pl.LightningDataModule): 

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
      num_workers = self._NUM_WORKERS, 
    )

    return train_data_loader 

  def val_data_loader(self): 

    val_dataset = pytorchvideo.data.Kinetics(
      data_path = os.path.join(self._DATA_PATH, "valid"),
      clip_sampler = pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
      decode_audio = False,
      transform = get_transform() 
    )

    val_data_loader = torch.utils.data.DataLoader(
      val_dataset, 
      batch_size = self._BATCH_SIZE, 
      num_workers = self._NUM_WORKERS 
    )
    return val_data_loader 


class VideoClassificationLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = make_kinetics_resnet() 
        #self.model = make_slowfast() 
        
    def forward(self, x): 
        return self.model(x) 
    
    def training_step(self, batch, batch_idx) : 
        # Input shape: (B,C,T,H,W)
        
        y_hat = self.model(batch["video"]) 
        loss = F.cross_entropy(y_hat, batch["label"]) 

        self.log("train_loss", loss.item()) 

        return loss 

    def validation_step(self, batch, batch_idx): 
        y_hat = self.model(batch["video"]) 
        loss = F.cross_entropy(y_hat, batch["label"]) 
        self.log("val_loss", loss.item())
        return loss 
    
    def configure_optimizers(self): 
        optimizer =  torch.optim.Adam(self.model.parameters(),lr=1e-1) 
        return optimizer 



def main():

    data_module = KineticsDataModule() 
    classification_module = VideoClassificationLightningModule() 
  
    trainer = pl.Trainer(gpus=1, max_epochs = 100, progress_bar_refresh_rate = 20, default_root_dir= '../../logs/05_21')  
    trainer.fit(classification_module, data_module.train_data_loader(), data_module.val_data_loader())  
 

if __name__== "__main__": 
    main() 