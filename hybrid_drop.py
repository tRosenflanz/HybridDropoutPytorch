import torch
import torch.nn as nn



class _HybridDropout_Base(nn.Module):
    def __init__(self,rate):
        super(_HybridDropout_Base, self).__init__()
        self.rate = rate
    def forward(self, x):
        if self.training:
            x = self.logic(x)
        return x



class HybridDropout_Spatial(_HybridDropout_Base):
    def __init__(self,rate):
        super(HybridDropout_Spatial, self).__init__(rate)
    
    def logic(self,x):
        assert len(x.shape)==4
        #compute e mask
        mask =torch.rand_like(x[:,[0]]).ge(torch.rand(x.size()).type(x.type())*self.rate)\
                   .float().expand_as(x)
        
        #compute U dot (random sample)
        u_dot=x[torch.randint(x.shape[0],size=(x.shape[0],),dtype=torch.long)]
        return x*mask + u_dot*(1-mask)

        
class HybridDropout_Normal(_HybridDropout_Base):
    def __init__(self,rate):
        super(HybridDropout_Normal, self).__init__(rate)
    
    
    def logic(self,x):
        #compute e mask
        mask =torch.rand_like(x).ge(torch.rand(x.size()).type(x.type())*self.rate).float()
        #compute U dot (random sample)
        u_dot=x[torch.randint(x.shape[0],size=(x.shape[0],),dtype=torch.long)]
        return x*mask + u_dot*(1-mask)