class WeakSupervisionDataloader():
    def __init__(self, gra_dataloader, seg_dataloader, recycling=True):
        self.gra_dataloader = gra_dataloader
        self.seg_dataloader = seg_dataloader
        if self.gra_dataloader is not None:
            self.batch_size = self.gra_dataloader.batch_size
        elif self.seg_dataloader is not None:
            self.batch_size = self.seg_dataloader.batch_size
        self.recycling = recycling  # True :若seg_data抽取完时， grade还没有抽取完，则循环抽取。  False:任意一方抽取完毕就结束

    def __iter__(self, ):
        if self.gra_dataloader is not None:
            self.gra_dataloader_iter = iter(self.gra_dataloader)
        if self.seg_dataloader is not None:
            self.seg_dataloader_iter = iter(self.seg_dataloader)
        return self

    def __next__(self):
        if self.gra_dataloader is not None and self.seg_dataloader is not None:   #联合监督
            imgG, label, imgG_path = next(self.gra_dataloader_iter)
            if self.recycling == True:
                try:
                    imgS, mask, Slabel, imgS_path = next(self.seg_dataloader_iter)
                except StopIteration:  # 由于分割样本要少，所以需要循环读取
                    self.seg_dataloader_iter = iter(self.seg_dataloader)
                    imgS, mask, Slabel, imgS_path = next(self.seg_dataloader_iter)
            else:
                imgS, mask, Slabel, imgS_path = next(self.seg_dataloader_iter)
            imgPath = imgG_path + imgS_path
        elif self.gra_dataloader is not None:   #只分类数据集
            imgG, label, imgG_path = next(self.gra_dataloader_iter)
            imgS = None
            mask = None
            Slabel = None
            imgS_path = None
        elif self.seg_dataloader is not None:    #只分割数据集
            imgS, mask, Slabel, imgS_path = next(self.seg_dataloader_iter)
            imgG = None
            label = None
            imgG_path = None

        return imgG, label, imgS, mask, Slabel, imgG_path, imgS_path

    def __len__(self):
        if self.gra_dataloader is not None:
            return len(self.gra_dataloader)
        else:
            return len(self.seg_dataloader)
