import numpy as np
import torch
from torch_geometric.data import Data
from torch_points3d.core.data_transform import SphereSampling
from torch_points3d.datasets.multimodal.image import ImageData
from torch_points3d.datasets.multimodal.forward_star import ForwardStar
from .projection import compute_index_map
from tqdm.auto import tqdm as tq



"""
Image-based transforms for multimodal data processing. Inspired by 
torch_points3d and torch_geometric transforms on data, with a signature 
allowing for multimodal transform composition : __call(data, images, mappings)__
"""



class NonStaticImageMask(object):
    """
    Transform-like structure. Find the mask of identical pixels accross a list
    of images.
    """
    def __init__(self, mask_size=(2048, 1024), n_sample=5):
        self.mask_size = tuple(mask_size)
        self.n_sample = n_sample


    def _process(self, images):
        images.map_size_high = self.mask_size
        images.mask = images.non_static_pixel_mask(size=self.mask_size, n_sample=self.n_sample)
        return images


    def __call__(self, data, images, mappings=None):
        """
        Compute the projection of data points into images and return the input 
        data augmented with attributes mapping points to pixels in provided 
        images.

        Expects Data (or anything), ImageData or List(ImageData), 
        ForwardStar mapping (or anything).

        Returns the same input. The mask is saved in ImageData attributes, to 
        be used for any subsequent image processing.
        """
        if isinstance(images, list):
            images = [self._process(img) for img in images]
        else:
            images = self._process(images)
        return data, images, mappings


    def __repr__(self):
        return self.__class__.__name__
    
#-------------------------------------------------------------------------------

class PointImagePixelMapping(object):
    """
    Transform-like structure. Computes the mappings between individual 3D points
    and image pixels. Point mappings are identified based on the self.key point
    identifiers.
    """
    def __init__(
            self,
            map_size_high=(2048, 1024),
            map_size_low=(512, 256),
            crop_top=0,
            crop_bottom=0,
            voxel=0.1,
            r_max=30,
            r_min=0.5,
            growth_k=0.2,
            growth_r=10,
            empty=0,
            no_id=-1,
            key='point_index'
        ):

        self.key = key
        self.empty = empty
        self.no_id = no_id

        # Store the projection parameters destined for the ImageData attributes.
        self.map_size_high = tuple(map_size_high)
        self.map_size_low = tuple(map_size_low)
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.voxel = voxel
        self.r_max = r_max
        self.r_min = r_min
        self.growth_k = growth_k
        self.growth_r = growth_r

  
    def _process(self, data, images):
        assert hasattr(data, self.key)
        assert isinstance(images, ImageData)
        assert images.num_images >= 1, "At least one image must be provided."

        # Pass the projection attributes to the ImageData
        images.map_size_high = self.map_size_high
        images.map_size_low = self.map_size_low
        images.crop_top = self.crop_top
        images.crop_bottom = self.crop_bottom
        images.voxel = self.voxel
        images.r_max = self.r_max
        images.r_min = self.r_min
        images.growth_k = self.growth_k
        images.growth_r = self.growth_r

        if images.mask is not None:
            assert images.mask.shape == images.map_size_high

        # Initialize the mapping arrays
        image_ids = []
        point_ids = []
        pixels = []

        # Project each image and gather the point-pixel mappings
        for i_image, image in tq(enumerate(images)):

            # Subsample the surrounding point cloud
            sampler = SphereSampling(image.r_max, image.pos, align_origin=False)
            data_sample = sampler(data.clone())

            # Projection to build the index map
            id_map, _ = compute_index_map(
                (data_sample.pos - image.pos.squeeze()).numpy(),
                getattr(data_sample, self.key).numpy(),
                np.array(image.opk.squeeze()),
                img_mask=image.mask.numpy() if image.mask is not None else None,
                img_size=image.map_size_high,
                crop_top=image.crop_top,
                crop_bottom=image.crop_bottom,
                voxel=image.voxel,
                r_max=image.r_max,
                r_min=image.r_min,
                growth_k=image.growth_k,
                growth_r=image.growth_r,
                empty=self.empty,
                no_id=self.no_id,
            )

            # Convert the id_map to id-xy coordinate soup
            # First column holds the point indices, subsequent columns hold the 
            # pixel coordinates. We use this heterogeneous soup to search for 
            # duplicate rows after resolution coarsening.
            # NB : no_id pixels are ignored
            active_pixels = np.where(id_map != self.no_id)
            point_ids_pixel_soup = id_map[active_pixels]
            point_ids_pixel_soup = np.column_stack((point_ids_pixel_soup,
                np.stack(active_pixels).transpose()))

            # Convert to lower resolution coordinates
            # NB: we assume the resolution ratio is the same for both 
            # dimensions 
            point_ids_pixel_soup[:, 1:] = image.coarsen_coordinates(point_ids_pixel_soup[:, 1:])

            # Remove duplicate id-xy in low resolution
            # Sort by point id
            point_ids_pixel_soup = np.unique(point_ids_pixel_soup, axis=0)  # bottleneck here ! Custom unique-sort with numba ?

            # Cast pixel coordinates to a dtype minimizing memory use
            point_ids_ = point_ids_pixel_soup[:, 0]
            pixels_ = np.asarray(torch.from_numpy(point_ids_pixel_soup[:,1:]).type(image.map_dtype))
            del point_ids_pixel_soup

            # Gather per-image mappings in list structures, only to be
            # numpy-stacked once all images are processed
            image_ids.append(i_image)
            point_ids.append(point_ids_)
            pixels.append(pixels_)
            del pixels_, point_ids_

        # Concatenate mappings
        image_ids = np.repeat(image_ids, [x.shape[0] for x in point_ids])
        point_ids = np.concatenate(point_ids)
        pixels = np.vstack(pixels)

        # Raise error if no point-image-pixel mapping was found
        if pixels.shape[0] == 0:
            raise ValueError("No mappings were found between the 3D points and any of the provided \
images. This will cause errors in the subsequent operations. Make sure your images are located in \
the vicinity of your point cloud and that the projection parameters allow for at least one \
point-image-pixel mapping before re-running this transformation.")

        # Sort by point_ids first, image_ids second
        sorting = np.lexsort((image_ids, point_ids))
        image_ids = image_ids[sorting]
        point_ids = point_ids[sorting]
        pixels = pixels[sorting]
        del sorting

        # We want all images present in the mappings and in ImageData to have 
        # been seen. If an image has not been seen, we remove it here.
        seen_image_ids = np.unique(image_ids)
        images = images[np.isin(np.arange(images.num_images), seen_image_ids)]
        image_ids = np.digitize(image_ids, seen_image_ids) - 1

        # Convert to "nested Forward Star" format
        # Compute image jumps in the pixels array
        image_pixel_mappings = ForwardStar(image_ids, pixels, dense=True)
        
        # Compress point_ids and image_ids by taking the last value of each jump
        image_ids = image_ids[image_pixel_mappings.jumps[1:] - 1]
        point_ids = point_ids[image_pixel_mappings.jumps[1:] - 1]

        # Compute point jumps in the image_ids array
        mappings = ForwardStar(point_ids, image_ids, image_pixel_mappings, dense=True,
            is_index_value=[True, False])

        # Compress point_ids by taking the last value of each jump
        point_ids = point_ids[mappings.jumps[1:] - 1]

        # Some points may have been seen by no image so we need to inject 
        # 0-sized jumps to account for these.
        # NB: we assume all relevant points are present in data.point_index, 
        #     if a point with an id larger than max(data.point_index) were to 
        #     exist, we would not be able to take it into account in the jumps.
        num_points = getattr(data, self.key).numpy().max() + 1
        mappings = mappings.reindex_groups(point_ids,
            num_groups=num_points)

        return data, images, mappings


    def __call__(self, data, images, mappings=None):
        """
        Compute the projection of data points into images and return the input 
        data augmented with attributes mapping points to pixels in provided 
        images.

        Expects a Data and a ImageData or a List(Data) and a List(ImageData) of 
        matching lengths.

        Returns the input data and the point-image-pixel mappings in a nested 
        ForwardStar format.
        """
        if isinstance(data, list):
            assert isinstance(images, list) and len(data) == len(images), \
                f"List(Data) items and List(ImageData) must have the same lengths."
            out = [self._process(d, i) for d, i in zip(data, images)]
            data, images, mappings = [list(x) for x in zip(*out)]

        else:
            data, images, mappings = self._process(data, images)

        return data, images, mappings


    def __repr__(self):
        return self.__class__.__name__

#-------------------------------------------------------------------------------

class PointImagePixelMappingFromId(object):
    """
    Transform-like structure. Intended to be called on _datas and _images_datas.

    Populate the passed Data sample in-place with attributes extracted from the 
    input ForwardStar mappings, based on the self.key point identifiers.
    
    The indices in data are expected to be included in those in mappings. The 
    ForwardStar format implicitly holds values for all self.key in 
    [0, ..., len(mappings)].
    """
    def __init__(self, key='point_index'):
        self.key = key


    def _process(self, data, images, mappings):
        assert isinstance(data, Data)
        assert hasattr(data, self.key)
        assert isinstance(images, ImageData)
        assert isinstance(mappings, ForwardStar)

        # Point indices to subselect mappings.
        # The selected mappings are sorted by their order in point_indices. 
        # NB: just like images, the same point may be used multiple times. 
        point_indices = torch.unique(data[self.key])
        mappings = mappings[point_indices]

        # Update point indices to the new mappings length.
        # This is important to preserve the mappings and for multimodal data
        # batching mechanisms.
        data[self.key] = torch.bucketize(data[self.key], point_indices)

        # Subselect the images used in the mappings.
        # The selected images are sorted by their order in image_indices.
        image_indices = np.unique(mappings.values[0])
        images = images[image_indices]

        # Update image indices to the new images length
        # This is important to preserve the mappings and for multimodal data
        # batching mechanisms.
        mappings.values[0] = np.digitize(mappings.values[0], image_indices) - 1

        return data, images, mappings


    def __call__(self, data, images, mappings):
        """
        Populate data sample in place with image attributes in mappings,
        based on the self.key point identifiers.
        """
        if isinstance(data, list):
            if isinstance(imagess, list) and isinstance(mappings, list) and \
                    len(images) == len(data) and len(mappings) == len(data):
                out = [self._process(d, i, m) for d, i, m in zip(data, images, mappings)]
            else:
                out = [self._process(d, images, mappings) for d in data]
            data, images, mappings = [list(x) for x in zip(*out)]

        else:
            data, images, mappings = self._process(data, images, mappings)

        return data, images, mappings


    def __repr__(self):
        return self.__class__.__name__

