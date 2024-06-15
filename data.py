import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance


def cv_random_flip(img, label,depth,edge):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, depth,edge
def randomCrop(image, label,depth,edge):
    border=20
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region),depth.crop(random_region),edge.crop(random_region)
def randomRotation(image,label,depth,edge):
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-15, 15)
        image=image.rotate(random_angle, mode)
        label=label.rotate(random_angle, mode)
        depth=depth.rotate(random_angle, mode)
        edge=edge.rotate(random_angle, mode)
    return image,label,depth,edge
def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image
def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im
    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))
def randomPeper(img):

    img=np.array(img)
    noiseNum=int(0.0015*img.shape[0]*img.shape[1])
    for i in range(noiseNum):

        randX=random.randint(0,img.shape[0]-1)  

        randY=random.randint(0,img.shape[1]-1)  

        if random.randint(0,1)==0:  

            img[randX,randY]=0  

        else:  

            img[randX,randY]=255 
    return Image.fromarray(img)  

# dataset for training
#The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
#(e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.

class SalObjDataset(data.Dataset):
    def __init__(self, root, trainsize):
        image_root = os.path.join(root, 'train_images/')
        gt_root = os.path.join(root, 'train_masks/')
        depth_root = os.path.join(root, 'train_depth/')
        edge_root = os.path.join(root, 'train_edges/')        
        
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')  or f.endswith('.bmp')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.depths=[depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                    or f.endswith('.png')  or f.endswith('.jpg')  or  f.endswith('.tiff')]

        self.edges=[edge_root + f for f in os.listdir(edge_root) if f.endswith('.bmp')
            or f.endswith('.png')  or f.endswith('.tiff')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths=sorted(self.depths)
        self.edges=sorted(self.edges)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])
        self.edge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
    def __getitem__(self, index):
        fname=self.images[index].split('/')[-1].split('.')[-2]
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth=self.binary_loader(self.depths[index])
        edge=self.binary_loader(self.edges[index])
        image,gt,depth,edge =cv_random_flip(image,gt,depth,edge)
        image,gt,depth,edge=randomCrop(image, gt,depth,edge)
        image,gt,depth,edge=randomRotation(image, gt,depth,edge)
        image=colorEnhance(image)
        # gt=randomGaussian(gt)
        gt=randomPeper(gt)
        image = self.img_transform(image) 
        gt = self.gt_transform(gt)   
        depth=self.depths_transform(depth).repeat(3,1,1)  
        edge=self.edge_transform(edge)
        return image, gt, depth,edge,fname

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts)==len(self.images)
        images = []
        gts = []
        depths=[]
        edges=[]
        for img_path, gt_path,depth_path,edge_path in zip(self.images, self.gts, self.depths,self.edges):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth= Image.open(depth_path)
            if img.size == gt.size and gt.size==depth.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
                edges.append(edge_path)
        self.images = images
        self.gts = gts
        self.depths=depths
        self.edges=edges
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth,edge):
        assert img.size == gt.size and gt.size==depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST),depth.resize((w, h), Image.NEAREST),edge.resize((w, h), Image.NEAREST)
        else:
            return img, gt, depth,edge

    def __len__(self):
        return self.size

    
class TestDataset(data.Dataset):
    def __init__(self, root, trainsize):
        image_root = os.path.join(root, 'RGB/')
        gt_root = os.path.join(root, 'gt/')
        depth_root = os.path.join(root, 'depth/')
        edge_root = os.path.join(root, 'edge/')
        
        
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')  or f.endswith('.bmp')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.depths=[depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                    or f.endswith('.png')  or f.endswith('.jpg')  or  f.endswith('.tiff')]
        
        self.edges=[edge_root + f for f in os.listdir(edge_root) if f.endswith('.bmp')
                    or f.endswith('.png')  or f.endswith('.jpg')  or  f.endswith('.tiff')]
                
        
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths=sorted(self.depths)
        self.edges=sorted(self.edges)
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.edge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        fname=self.images[index].split('/')[-1].split('.')[-2]
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth=self.binary_loader(self.depths[index])
        edge=self.binary_loader(self.edges[index])
        image = self.img_transform(image) 
        gt = self.gt_transform(gt)   
        depth=self.depths_transform(depth).repeat(3,1,1)   
        edge=self.edge_transform(edge)
        return image, gt, depth,edge,fname
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth,edge):
        assert img.size == gt.size and gt.size==depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST),depth.resize((w, h), Image.NEAREST),edge.resize((w, h), Image.NEAREST)
        else:
            return img, gt, depth,edge

    def __len__(self):
        return self.size   