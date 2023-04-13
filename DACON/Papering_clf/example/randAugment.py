from timm.data.auto_augment import rand_augment_transform, RandAugment
from PIL import Image
from matplotlib import pyplot as plt

tfm = rand_augment_transform(
    config_str='rand-m9-mstd0.5', 
    hparams={'translate_const': 117, 'img_mean': (124, 116, 104)}
)

x   = Image.open("../../imagenette2-320/train/n01440764/ILSVRC2012_val_00000293.JPEG")
plt.imshow(x)

plt.imshow(tfm(x))