import torch.optim as optim 
import torch.nn as nn 
import torch.nn.functional as F 
from create_model import make_kinetics_resnet 
import pytorch_lightning 
import torch 

class VideoClassificationLightningModule(pytorch_lightning.LightningModule): 
    def __init__(self):
        super().__init__()
        self.model = make_kinetics_resnet() 

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
        return torch.optim.Adam(self.model.parameters(),lr=1e-1) 



