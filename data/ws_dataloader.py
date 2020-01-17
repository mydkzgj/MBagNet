from torch.utils.data import DataLoader

class WeakSupervisionDataloader():
    def __init__(self, gra_dataloader, seg_dataloader):
        self.gra_dataloader = gra_dataloader
        self.seg_dataloader = seg_dataloader
        self.batch_size = self.gra_dataloader.batch_size
    def __iter__(self, ):
        self.gra_dataloader_iter = iter(self.gra_dataloader)
        self.seg_dataloader_iter = iter(self.seg_dataloader)
        return self

    def __next__(self):
        imgG, label,  = next(self.gra_dataloader_iter)
        #imgS, mask, = next(self.seg_dataloader_iter)
        #"""
        try:
            imgS, mask, Slabel = next(self.seg_dataloader_iter)
        except StopIteration:     #由于分割样本要少，所以需要循环读取            
            self.seg_dataloader_iter = iter(self.seg_dataloader)
            imgS, mask, Slabel = next(self.seg_dataloader_iter)
        #"""
        return imgG, label, imgS, mask, Slabel

    def __len__(self):
        return len(self.gra_dataloader)
