from PIL import Image
from timm.data.random_erasing import RandomErasing
from torchvision import transforms
from matplotlib import pyplot as plt

img = Image.open("../../imagenette2-320/train/n01440764/ILSVRC2012_val_00000293.JPEG")
x   = transforms.ToTensor()(img)
plt.imshow(x.permute(1, 2, 0))

random_erase = RandomErasing(probability=1, mode='pixel', device='cpu')
plt.imshow(random_erase(x).permute(1, 2, 0))


from timm.data.random_erasing import RandomErasing

# get input images and convert to `torch.tensor`
X, y = input_training_batch()
X = convert_to_torch_tensor(X)

# perform RandomErase data augmentation
random_erase = RandomErasing(probability=0.5)

# get augmented batch
X_aug = random_erase(X)

# do something here