import numpy as np
import os
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
from src.sample_tiles import *

### SAMPLE TRIPLETS
img_dir = 'data/images' # directory where images are saved
img_triplets = get_triplet_imgs(img_dir, n_triplets=20) # This means we need 40 images
print(img_triplets[:5,:])

tile_dir = 'data/example_tiles2' # where you want to save your tiles
tiles = get_triplet_tiles(tile_dir,
                          img_dir,
                          img_triplets,
                          tile_size=50,
                          val_type='uint8',
                          bands_only=True,
                          save=True,
                          verbose=True)
tile_dir = 'data/example_tiles2/'
n_triplets = 2
plt.rcParams['figure.figsize'] = (12,4)
for i in range(n_triplets):
  tile = np.load(os.path.join(tile_dir, str(i)+'anchor.npy'))
  neighbor = np.load(os.path.join(tile_dir, str(i)+'neighbor.npy'))
  distant = np.load(os.path.join(tile_dir, str(i)+'distant.npy'))

  vmin = np.array([tile, neighbor, distant]).min()
  vmax = np.array([tile, neighbor, distant]).max()

  plt.figure()
  plt.subplot(1,3,1)
  plt.imshow(tile[:,:,[0,1,2]])
  plt.title('Anchor '+str(i))
  plt.subplot(1,3,2)
  plt.imshow(neighbor[:,:,[0,1,2]])
  plt.title('Neighbor '+str(i))
  plt.subplot(1,3,3)
  plt.imshow(distant[:,:,[0,1,2]])
  plt.title('Distant '+str(i))
  plt.show()
