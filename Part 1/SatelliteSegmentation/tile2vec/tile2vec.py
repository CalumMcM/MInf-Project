import sys
import os
import torch
from torch import optim
from time import time

tile2vec_dir = '/atlas/u/swang/software/GitHub/tile2vec'
sys.path.append('../')
sys.path.append(tile2vec_dir)

from src.datasets import TileTripletsDataset, GetBands, RandomFlipAndRotate, ClipAndScale, ToFloatTensor, triplet_dataloader
from src.tilenet import make_tilenet
from src.training import prep_triplets, train_triplet_epoch

# Environment stuff
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda = torch.cuda.is_available()

# Change these arguments to match your directory and desired parameters
img_type = 'naip'
tile_dir = '/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/tile2vec/data/toa_triplets'
#tile_dir = '/Users/calummcmeekin/Downloads/tile2vec-master-original/data/triplets/'
bands = 3
augment = True
batch_size = 50
shuffle = True
num_workers = 4
n_triplets = 58#202

dataloader = triplet_dataloader(img_type, tile_dir, bands=bands, augment=augment,
                                batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                n_triplets=n_triplets, pairs_only=True)
print('Dataloader set up complete.')



# Set up TileNet

in_channels = bands
z_dim = 512

TileNet = make_tilenet(in_channels=in_channels, z_dim=z_dim)
TileNet.train()
if cuda: TileNet.cuda()
print('TileNet set up complete.')

lr = 1e-3
optimizer = optim.Adam(TileNet.parameters(), lr=lr, betas=(0.5, 0.999))


# TRAIN MODEL

epochs = 1
margin = 10
l2 = 0.01
print_every = 1000
save_models = True

model_dir = '/Users/calummcmeekin/Documents/GitHub/MInf-Project/SatelliteSegmentation/tile2vec/models'
if not os.path.exists(model_dir): os.makedirs(model_dir)

t0 = time()
with open("results_fn", 'w') as file:

    print('Begin training.................')
    for epoch in range(0, epochs):
        (avg_loss, avg_l_n, avg_l_d, avg_l_nd) = train_triplet_epoch(
            TileNet, cuda, dataloader, optimizer, epoch+1, margin=margin, l2=l2,
            print_every=print_every, t0=t0)

# Save model after last epoch
if save_models:
    model_fn = os.path.join(model_dir, 'TileNet_epoch100_toa_data.ckpt')
    torch.save(TileNet.state_dict(), model_fn)
