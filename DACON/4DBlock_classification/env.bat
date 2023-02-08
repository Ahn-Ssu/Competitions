conda create -n Vision -y

conda update conda --all -y 

conda install numpy -y

conda install pandas -y

conda install scipy -y

conda install matplotlib -y

conda install -c conda-forge scikit-learn -y

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

# [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) 

conda install -c conda-forge opencv -y

conda install -c conda-forge imgaug -y

conda install -c conda-forge albumentations -y

conda install -c conda-forge tqdm -y 

conda install -c conda-forge umap-learn -y

