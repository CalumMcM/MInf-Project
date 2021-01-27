import numpy as np
import gdal
import os
import random
import rasterio as rs
import re
import subprocess

def load_img(img_file, val_type='uint8', bands_only=False, num_bands=4):
    """
    Loads an image using gdal, returns it as an array.
    """
    obj = gdal.Open(img_file)
    if val_type == 'uint8':
        img = obj.ReadAsArray().astype(np.uint8)
    elif val_type == 'float32':
        img = obj.ReadAsArray().astype(np.float32)
    else:
        raise ValueError('Invalid val_type for image values. Try uint8 or float32.')
    img = np.moveaxis(img, 0, -1)
    if bands_only: img = img[:,:,:num_bands]
    return img

def load_rs_img(img_file, val_type='unit8', bands_only=False, num_bands=3):
    """
    Will open an image using rasterio
    """

    # Open the iamge
    img = rs.open(img_file)

    # Print number of bands in the image:
    img = img.read([1,2,3])
    
    return img
    
def get_triplet_imgs(img_dir, img_ext='.tif', n_triplets=1000):
    """
    Returns a numpy array of dimension (n_triplets, 2). First column is
    the img name of anchor/neighbor tiles and second column is img name 
    of distant tiles.
    """
    img_names = []
    for filename in os.listdir(img_dir):
        if filename.endswith(img_ext):
            img_names.append(filename)
    img_triplets = list(map(lambda _: random.choice(img_names), range(2 * n_triplets)))
    img_triplets = np.array(img_triplets)
    return img_triplets.reshape((-1, 2))

def get_triplet_imgs_simple(img_dir, img_ext='.tif', n_triplets=1000):
    """
    Returns a numpy array of images in a given directory
    """
    img_names = []
    for filename in os.listdir(img_dir):
        if filename.endswith(img_ext):
            img_names.append(filename)
    #img_triplets = list(map(lambda _: random.choice(img_names), range(2 * n_triplets)))
    #img_triplets = np.array(img_triplets)
    #return img_triplets.reshape((-1, 2))
    return img_names

def get_triplet_imgs_quad(biome_name, quads=[1,2,3,4], img_ext='.tif'):
    """
    Returns a numpy 2D array where dimension 1 is the quadrant and dimension
    2 is the images for that quadrant
    """
    biome_quads = []
    for quad in quads:
        
        img_names = []
        
        # Construct path to current quad folder
        path = os.path.join('../data', biome_name + " Quads")
        cur_quad_foldr_name = 'Quad ' + str(quad)
        path = os.path.join(path, cur_quad_foldr_name)
        
        # Go through each file in current quad and append 
        # chosen images to array
        
        for filename in os.listdir(path):
            if filename.endswith(img_ext):
                img_names.append(filename)
                
        # Append quad imgs to biome array
        biome_quads.append(np.unique(img_names))
        
    return biome_quads


def get_triplet_tiles_simple(tile_dir, img_dir, amazonia_quads, cerrado_quads, caatinga_quads, training_quads = [1,2,3], num_triplets_per_biome=10, seed = 0, val_type='uint8', bands_only=False, save=True, verbose=False):
    
    np.random.seed(0)
    
    if not os.path.exists(tile_dir):
        os.makedirs(tile_dir)
    
    cur_triplet_num = 0
    
    for anchor in ['Amazonia', 'Cerrado', 'Caatinga']:
        
        print ("\n\nCURRENT BIOME: {}".format(anchor))
        # Go through each quadrant generating anchor, neighbour and distant 
        # tripletls
        for quad in training_quads:
            
            print ("\n\nCURRENT QUAD: {}\n\n".format(quad))
            
            anchor_dir = os.path.join(img_dir, anchor + " Quads", 'Quad '+str(quad))
            
            if (anchor == 'Amazonia'):

                # Get unique anchor images
                unique_anchor_imgs = amazonia_quads[quad-1]

                # Get unique neighbour images
                unique_neighbour_imgs = amazonia_quads[quad-1]

                # Get distant neighbour images
                unique_distant_imgs_1 = cerrado_quads
                unique_distant_imgs_2 = caatinga_quads

                # Set distant image directories
                dist_dir_1 = os.path.join(img_dir, 'Cerrado Quads')
                dist_dir_2 = os.path.join(img_dir, 'Caatinga Quads')

            elif (anchor == 'Cerrado'):

                # Get unique anchor images
                unique_anchor_imgs = cerrado_quads[quad-1]

                # Get unique neighbour images
                unique_neighbour_imgs = cerrado_quads[quad-1]

                # Get distant neighbour images
                unique_distant_imgs_1 = amazonia_quads
                unique_distant_imgs_2 = caatinga_quads

                # Set distant image directories
                dist_dir_1 = os.path.join(img_dir, 'Amazonia Quads')
                dist_dir_2 = os.path.join(img_dir, 'Caatinga Quads')

            elif (anchor == 'Caatinga'):

                # Get unique anchor images
                unique_anchor_imgs = caatinga_quads[quad-1]

                # Get unique neighbour images
                unique_neighbour_imgs = caatinga_quads[quad-1]

                # Get distant neighbour images
                unique_distant_imgs_1 = amazonia_quads
                unique_distant_imgs_2 = cerrado_quads

                # Set distant image directories
                dist_dir_1 = os.path.join(img_dir, 'Amazonia Quads')
                dist_dir_2 = os.path.join(img_dir, 'Cerrado Quads')

            biome_triplet_counter = 0
            
            for anchor_img_name in unique_anchor_imgs:
    
                if biome_triplet_counter >= num_triplets_per_biome: break

                if verbose:
                    print("Sampling image {} for {} biome".format(anchor_img_name, anchor))

                if anchor_img_name[-3:] == 'npy':
                    img = np.load(anchor_img_name)
                else:

                    image_in = os.path.join(anchor_dir, anchor_img_name)

                    # Load anchor image 
                    anchor_img = load_rs_img(image_in, val_type=val_type, 
                               bands_only=bands_only)

                    # Get neighbour image name
                    neighbour_img_name, unique_neighbour_imgs = get_random_image(anchor_img_name, unique_neighbour_imgs)
    
                    image_in = os.path.join(anchor_dir, neighbour_img_name)
            
                    # Load neighbour image
                    neighbour_img = load_rs_img(image_in, val_type=val_type, 
                               bands_only=bands_only)

                    # Get distant image
                    distant_img_name, distant_dir, unique_distant_imgs_1, unique_distant_imgs_2 = get_random_distant_image(unique_distant_imgs_1, unique_distant_imgs_2, dist_dir_1, dist_dir_2, len(training_quads))


                    image_in = os.path.join(distant_dir, distant_img_name)


                    # Load distant image
                    distant_img = load_rs_img(image_in, val_type=val_type, 
                               bands_only=bands_only)

                    # Save triplet images as numpy arrays
                    if verbose:
                            print("    Saving anchor, neighbor and distant tile #{} for biome {}".format(cur_triplet_num, anchor))
                            print ("Anchor: {}\nNeighbour: {}\nDistant: {}".format(anchor_img_name, neighbour_img_name, distant_img_name))

                    if save:

                        tile_anchor =  np.swapaxes(anchor_img, 0, 2)#np.array(anchor_img)
                        tile_neighbour =  np.swapaxes(neighbour_img, 0, 2)#np.array(neighbour_img)
                        tile_distant =  np.swapaxes(distant_img, 0, 2)#np.array(distant_img)

                        tile_anchor_1 =  reset_shape(tile_anchor)
                        tile_neighbour_1 =  reset_shape(tile_neighbour)
                        tile_distant_1 = reset_shape(tile_distant)

                        tile_anchor_2 =  remove_nan(reset_shape(tile_anchor))
                        tile_neighbour_2 =  remove_nan(reset_shape(tile_neighbour))
                        tile_distant_2 = remove_nan(reset_shape(tile_distant))

                        np.save(os.path.join(tile_dir, '{}anchor.npy'.format(cur_triplet_num)), tile_anchor_2)
                        np.save(os.path.join(tile_dir, '{}neighbor.npy'.format(cur_triplet_num)), tile_neighbour_2)
                        np.save(os.path.join(tile_dir, '{}distant.npy'.format(cur_triplet_num)), tile_distant_2)

                        cur_triplet_num += 1
                        biome_triplet_counter += 1

                        print ("BIOME {} progress: {:.2f}%".format(anchor, (biome_triplet_counter/len(unique_anchor_imgs)*100)))

def reset_shape(tile):
    """
    Takes a tile and removes/pads it
    so that the returned tile has shape
    (51, 51, 3)
    """
    x, y, z = tile.shape
        
    # Reduce shape
    if (x == 51 and y == 51):

        return tile
    
    elif (x>51):
        tile = np.delete(tile, -1, 0)
        return reset_shape(tile)
    
    elif (y>51):
        tile = np.delete(tile, -1, 1)
        return reset_shape(tile)

    # Pad shape
    elif (x<51):

        basic = np.array([tile[0]])

        tile = np.vstack((tile, basic))
        
        return reset_shape(tile)
    
    elif (y < 51):
                
        mean = np.mean(tile)
        new_col = np.full((51,1,3), mean)

        tile = np.append(tile, new_col, axis=1)
        
        #tile = np.append(tile, basic, axis=1)
                
        return reset_shape(tile)

def remove_nan(new_tile):
    """
    Takes a tile and replaces any nan values with the mean
    of the image the nan appears in
    """

    mean = np.nanmean(new_tile)
    
    new_tile = np.nan_to_num(new_tile, nan=mean, posinf=mean, neginf=mean)
    
    return new_tile

def get_random_image(given_image, unique_imgs_array):
    """
    Randomly selects an image from an array of images that is not the given image, 
    removes it from the array of images and returns that image name along 
    with the updated image array. This is to prevent the same image 
    being used twice. 
    """
    
    new_image = given_image
        
    while (new_image == given_image):
        new_image = random.sample(list(unique_imgs_array), k=1)[0]
    
    unique_imgs_array=np.delete(unique_imgs_array, np.where(unique_imgs_array == new_image))
    
    return new_image, unique_imgs_array

def get_random_distant_image(quads_1, quads_2, quads_dir_1, quads_dir_2, num_quads):
    """
    Randomly selects an image, from one of two randomly selected arrays,
    removes it from the array of images and returns that image name along 
    with the updated image array. This is to prevent the same image 
    being used twice. 
    """
    
    # Select distant biome    
    selected_array = random.randint(1,2)

    if (selected_array == 1):
    
        # Select quadrant from distant biome
        rad_quad = random.randint(0, num_quads-1)

        while (len(quads_1[rad_quad]) == 0):
            # There should never be the case when all 
            # quads are empty
            rad_quad = random.randint(1, num_quads)
            
        new_image = random.sample(list(quads_1[rad_quad]), k=1)[0]
        
        quads_1[rad_quad] = np.delete(quads_1[rad_quad], np.where(quads_1[rad_quad] == new_image))

        image_dir = os.path.join(quads_dir_1, "Quad " + str(rad_quad+1))

    else:
        
        # Select quadrant from distant biome
        rad_quad = random.randint(0, num_quads-1)
        
        while (len(quads_1[rad_quad]) == 0):
            # There should never be the case when all 
            # quads are empty
            rad_quad = random.randint(1, num_quads)
            
        new_image = random.sample(list(quads_2[rad_quad]), k=1)[0]
                
        quads_2[rad_quad] = np.delete(quads_2[rad_quad], np.where(quads_2[rad_quad] == new_image)) 
                
        image_dir = os.path.join(quads_dir_1, "Quad " + str(rad_quad+1))
        
    return new_image, image_dir, quads_1, quads_2
    
def get_triplet_tiles(tile_dir, img_dir, img_triplets, tile_size=50, neighborhood=100, 
                      val_type='uint8', bands_only=False, save=True, verbose=False):
    if not os.path.exists(tile_dir):
        os.makedirs(tile_dir)
    size_even = (tile_size % 2 == 0)
    tile_radius = tile_size // 2

    n_triplets = img_triplets.shape[0]
    unique_imgs = np.unique(img_triplets)
    tiles = np.zeros((n_triplets, 3, 2), dtype=np.int16)

    for img_name in unique_imgs:
        print("Sampling image {}".format(img_name))
        if img_name[-3:] == 'npy':
            img = np.load(img_name)
        else:
            img = load_img(os.path.join(img_dir, img_name), val_type=val_type, 
                       bands_only=bands_only)
            print (img.shape)
            img = load_rs_img(os.path.join(img_dir, img_name),val_type=val_type, 
                       bands_only=bands_only)
            img = np.swapaxes(img, 0, 2)
            print (img.shape)
            
            
            
        img_padded = np.pad(img, pad_width=[(tile_radius, tile_radius),
                                            (tile_radius, tile_radius), (0,0)],
                            mode='reflect')
        img_shape = img_padded.shape

        for idx, row in enumerate(img_triplets):
            if row[0] == img_name:
                xa, ya = sample_anchor(img_shape, tile_radius)
                xn, yn = sample_neighbor(img_shape, xa, ya, neighborhood, tile_radius)
                
                if verbose:
                    print("    Saving anchor and neighbor tile #{}".format(idx))
                    print("    Anchor tile center:{}".format((xa, ya)))
                    print("    Neighbor tile center:{}".format((xn, yn)))
                if save:
                    tile_anchor = extract_tile(img_padded, xa, ya, tile_radius)
                    tile_neighbor = extract_tile(img_padded, xn, yn, tile_radius)
                    if size_even:
                        tile_anchor = tile_anchor[:-1,:-1]
                        tile_neighbor = tile_neighbor[:-1,:-1]
                    np.save(os.path.join(tile_dir, '{}anchor.npy'.format(idx)), tile_anchor)
                    np.save(os.path.join(tile_dir, '{}neighbor.npy'.format(idx)), tile_neighbor)
                
                tiles[idx,0,:] = xa - tile_radius, ya - tile_radius
                tiles[idx,1,:] = xn - tile_radius, yn - tile_radius
                
                if row[1] == img_name:
                    # distant image is same as anchor/neighbor image
                    xd, yd = sample_distant_same(img_shape, xa, ya, neighborhood, tile_radius)
                    if verbose:
                        print("    Saving distant tile #{}".format(idx))
                        print("    Distant tile center:{}".format((xd, yd)))
                    if save:
                        tile_distant = extract_tile(img_padded, xd, yd, tile_radius)
                        if size_even:
                            tile_distant = tile_distant[:-1,:-1]
                        np.save(os.path.join(tile_dir, '{}distant.npy'.format(idx)), tile_distant)
                    tiles[idx,2,:] = xd - tile_radius, yd - tile_radius
            
            elif row[1] == img_name: 
                # distant image is different from anchor/neighbor image
                xd, yd = sample_distant_diff(img_shape, tile_radius)
                if verbose:
                        print("    Saving distant tile #{}".format(idx))
                        print("    Distant tile center:{}".format((xd, yd)))
                if save:
                    tile_distant = extract_tile(img_padded, xd, yd, tile_radius)
                    if size_even:
                        tile_distant = tile_distant[:-1,:-1]
                    np.save(os.path.join(tile_dir, '{}distant.npy'.format(idx)), tile_distant)
                tiles[idx,2,:] = xd - tile_radius, yd - tile_radius
            
    return tiles

def sample_anchor(img_shape, tile_radius):
    w_padded, h_padded, c = img_shape
    w = w_padded - 2 * tile_radius
    h = h_padded - 2 * tile_radius
    
    xa = np.random.randint(0, w) + tile_radius
    ya = np.random.randint(0, h) + tile_radius
    return xa, ya

def sample_neighbor(img_shape, xa, ya, neighborhood, tile_radius):
    w_padded, h_padded, c = img_shape
    w = w_padded - 2 * tile_radius
    h = h_padded - 2 * tile_radius
    
    xn = np.random.randint(max(xa-neighborhood, tile_radius),
                           min(xa+neighborhood, w+tile_radius))
    yn = np.random.randint(max(ya-neighborhood, tile_radius),
                           min(ya+neighborhood, h+tile_radius))
    return xn, yn


def sample_distant_same(img_shape, xa, ya, neighborhood, tile_radius):
    w_padded, h_padded, c = img_shape
    w = w_padded - 2 * tile_radius
    h = h_padded - 2 * tile_radius
    
    xd, yd = xa, ya
    while (xd >= xa - neighborhood) and (xd <= xa + neighborhood):
        xd = np.random.randint(0, w) + tile_radius
    while (yd >= ya - neighborhood) and (yd <= ya + neighborhood):
        yd = np.random.randint(0, h) + tile_radius
    return xd, yd


def sample_distant_diff(img_shape, tile_radius):
    return sample_anchor(img_shape, tile_radius)

def extract_tile(img_padded, x0, y0, tile_radius):
    """
    Extracts a tile from a (padded) image given the row and column of
    the center pixel and the tile size. E.g., if the tile
    size is 15 pixels per side, then the tile radius should be 7.
    """
    w_padded, h_padded, c = img_padded.shape
    row_min = x0 - tile_radius
    row_max = x0 + tile_radius
    col_min = y0 - tile_radius
    col_max = y0 + tile_radius
    assert row_min >= 0, 'Row min: {}'.format(row_min)
    assert row_max <= w_padded, 'Row max: {}'.format(row_max)
    assert col_min >= 0, 'Col min: {}'.format(col_min)
    assert col_max <= h_padded, 'Col max: {}'.format(col_max)
    tile = img_padded[row_min:row_max+1, col_min:col_max+1, :]
    return tile

