import torch
import numpy as np
import scipy

def compute_bures(AA, BB, max_iter=100):
   
    eps = np.finfo('float32').eps
   
    AAt = torch.tensor(AA.astype('float32'))
    BBt = torch.tensor(BB.astype('float32'))
   
    w = torch.randn(AA.shape[0],1, requires_grad=True)
    w.data = torch.sign(torch.sum(AAt,axis=0) - torch.sum(BBt,axis=0)).t()
    w.data = w / torch.sqrt(w.t() @ w)
   
    opt = torch.optim.Adam([w], lr=1e-2, betas=(0.9,0.999), eps=1e-8)
    #opt = torch.optim.SGD([w], lr=1e-3)
   
    for ii in range(max_iter):
       
        old_w = w.clone().detach()
       
        opt.zero_grad()
       
        WB = torch.sqrt(w.t() @ BBt @ w)
        WA = torch.sqrt(w.t() @ AAt @ w)
       
        obj = -( (WB - WA) / torch.sqrt(eps + w.t() @ w))
        obj.backward()
        opt.step()
       
        diff = torch.abs(w.detach() - old_w)
        #print('-------------------------------------------')
        #print('weight norm: ', (torch.sqrt(eps + w.t() @ w)))
        #print('obj: ', -obj)
       
    return obj.detach().cpu().numpy(), w.detach().cpu().numpy()
