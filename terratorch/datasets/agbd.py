import h5py
import torch
from torchgeo.datasets import NonGeoDataset
import numpy as np
import pickle
from os.path import join, exists
from datetime import datetime, timedelta
from astropy.time import Time
from astropy.coordinates import get_sun, EarthLocation, AltAz
from astropy import units

from pin_requirements import updated_deps_str

np.seterr(divide='ignore')
import pandas as pd
from tqdm import tqdm


# Define the nodata values for each data source
NODATAVALS = {
    'S2_bands': 0,
    'CH': 255,
    'ALOS_bands': 0,
    'DEM': -9999,
    'LC': 255
}

# Define the biomes
REF_BIOMES = {
    20: 'Shrubs',
    30: 'Herbaceous vegetation',
    40: 'Cultivated',
    90: 'Herbaceous wetland',
    111: 'Closed-ENL',
    112: 'Closed-EBL',
    114: 'Closed-DBL',
    115: 'Closed-mixed',
    116: 'Closed-other',
    121: 'Open-ENL',
    122: 'Open-EBL',
    124: 'Open-DBL',
    125: 'Open-mixed',
    126: 'Open-other'
}
BIOMES_KEY2IDX = {v: i for i, v in enumerate(REF_BIOMES.keys())}
BIOMES_KEYS = [v for v in REF_BIOMES.keys()]

# Define the start date of the GEDI mission
GEDI_START_MISSION = '2019-04-17'


class AGBD(NonGeoDataset):
    """Dataset class for [AGBD](https://arxiv.org/abs/2406.04928):
    above-ground biomass regression from multimodal remote sensing data.

    Dataset quick description
    # TODO

    Dataset Format
    # TODO

    Dataset Features
    # TODO

    Download instructions
    # TODO

    This implementation is adapted from the
    [official implementation](https://github.com/ghjuliasialelli/AGBD).

    If you use this dataset in your research, please cite the following
    paper:
    https://arxiv.org/abs/2406.04928
    """

    MODES = [
        'train',
        'val',
        'test',
    ]
    NORMALIZATION_STRATS = [
        'pct',
        'min_max',
        'mean_std',
    ]
    LANDCOVER_ENCODINGS = [
        'sincos',
        'onehot',
        'distribution',
        'cat2vec',
    ]

    def __init__(
            self,
            data_root: str,
            mode: str,
            use_latlon: bool = False,
            use_s2_bands: list[str] | bool = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'],
            use_canopy_height: bool = False,
            use_s1: bool = False,
            use_alos: bool = False,
            use_landcover: bool = False,
            landcover_encoding: str = 'sincos',
            use_dem: bool = False,
            use_gedi_dates: bool = False,
            use_s2_day: bool = False,
            use_s2_doy: bool = False,
            use_s2_sun: bool = False,
            use_topo_aspect: bool = False,
            use_topo_slope: bool = False,
            use_metadata: bool = False,
            normalization_strategy: str = 'pct',
            normalize_target: bool = False,
            patch_size: tuple[int, int] | list[int] = [25, 25],
            version: int = 4,
            years: list[int] | set[int] = [2019, 2020],
            mini: int = -1,
            chunk_size: int = 1,
    ):
        """

        Args:
            data_root (str): Path to the root directory for the AGBD
                dataset.
            chunk_size (int): Internal chunk size of the hdf5.
            mode (str):
            use_latlon (bool): Whether to include `lon_1` and `lon_2` in
                the input features. Default: False
            use_s2_bands (list[str] | bool): List of Sentinel-2 bands
                (e.g., `['B12']`) to consider as input for the model.
                Default: ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
            use_canopy_height (bool): Whether to include the canopy
                height predictions `ch` and `ch_std` patches in the
                input. Predictions were produced using Lang et al 2023
                https://www.nature.com/articles/s41559-023-02206-6
                Default: False
            use_s1 (bool): Whether to include the Sentinel-1 patches in
                the input.
                Default: False
            use_alos (bool): Whether to include the ALOS patches in the
                input.
                Default: False
            use_landcover (bool): Whether to include the biome landcover
                patches in the input.
                Default: False
            landcover_encoding (str): Encoding strategy for
                preprocessing the biome landcover classes. Must be one
                of `['sincos', 'onehot', 'distribution', 'cat2vec']`
                Default: 'sincos'
            use_dem (bool): Whether to include the Digital Elevation
                Model patches in the input.
                Default: False
            use_gedi_dates (bool): Whether to include the GEDI dates in
                the input.
                Default: False
            use_s2_day (bool): Whether to include the S2 date in the
                input.
                Default: False
            use_s2_doy (bool): Whether to include the S2 DOY in the
                input.
                Default: False
            use_s2_sun (bool): Whether to include the S2 sun position in
                the input.
                Default: False
            use_topo_aspect (bool): Whether to include the topological
                aspect.
                Default: False
            use_topo_slope (bool): Whether to include the topological
                slope.
                Default: False
            use_metadata (bool): Whether to return metadata
                (coordinates and date).
                Default: False
            patch_size (tuple[int, int] | list[int]): Size of the
                patches to extract, in pixels.
                Default: [25, 25]
            normalization_strategy (str): Normalization strategy for
                input data. Must be one of
                `['pct', 'min_max', 'mean_std']`.
                Default: 'pct'
            normalize_target (bool): Whether to normalize the target
                aboveground biomass estimates from GEDI.
                Default: False
            version (int): Dataset version.
                Default: 4
            years (list[int] | set[int]): Years of the dataset to
                consider.
                Default: `[2019, 2020]`
            mini (int): Number of files to load for each year. This
                allows working on a smaller subset of the dataset, for
                debugging of fast experimentation. Note that you can
                also reduce the dataset size by adjusting the `years`
                parameter.
                Default: -1
        """
        super().__init__()

        # Sanity checks
        assert mode in self.MODES, f"The mode must be one of {self.MODES}"
        assert patch_size[0] == patch_size[1], "The patch size must be square"
        assert normalization_strategy in self.NORMALIZATION_STRATS, \
            (f"The normalization strategy must be one of "
             f"{self.NORMALIZATION_STRATS}")
        assert landcover_encoding in self.LANDCOVER_ENCODINGS, \
            (f"The landcover encoding strategy must be one of "
             f"{self.LANDCOVER_ENCODINGS}")

        # Get the parameters
        self.h5_path = data_root
        self.norm_path = data_root
        self.mapping_path = data_root
        self.cat2vec_path = join(data_root, 'cat2vec')
        self.s2_tile_to_dates_path = data_root
        self.mode = mode
        self.chunk_size = chunk_size
        self.years = list(years)

        # Get the file names
        self.fnames = []
        for year in self.years:
            num_files = 20 if mini < 1 else min(mini, 20)
            self.fnames += [
                f'data_subset-{year}-v{version}_{i}-20.h5'
                for i in range(num_files)
            ]

        # Initialize the index
        self.index, self.length = initialize_index(
            self.fnames,
            self.mode,
            self.chunk_size,
            self.mapping_path,
            self.h5_path
        )

        # Define the data to use
        self.use_latlon = use_latlon
        self.use_s2_bands = use_s2_bands
        self.use_canopy_height = use_canopy_height
        self.use_s1 = use_s1
        self.use_alos = use_alos
        self.use_landcover = use_landcover
        self.landcover_encoding = landcover_encoding
        self.use_dem = use_dem
        self.use_gedi_dates = use_gedi_dates
        self.use_s2_day = use_s2_day
        self.use_s2_doy = use_s2_doy
        self.use_s2_sun = use_s2_sun
        self.use_topo_aspect = use_topo_aspect
        self.use_topo_slope = use_topo_slope
        self.use_metadata = use_metadata
        self.patch_size = patch_size

        # Define the learning procedure
        self.normalization_strategy = normalization_strategy
        self.normalize_target = normalize_target

        # Load the normalization values
        stats_file = f"statistics_subset_2019-2020-v{version}_new.pkl"
        if not exists(join(self.norm_path, stats_file)):
            raise FileNotFoundError(
                f"The file `{stats_file}` does not exist.")
        with open(join(self.norm_path, stats_file), mode='rb') as f:
            self.norm_values = pickle.load(f)

        # Open the file handles
        self.handles = {
            fname: h5py.File(join(self.h5_path, fname), 'r+')
            for fname in self.index.keys()
        }

        # Read the dictionary mapping S2 tiles and acquisition dates to
        # UTC times. This will be used for recovering the sun position
        self.s2_tile_to_dates = pickle.load(open(
            join(self.s2_tile_to_dates_path, "s2_tile_to_dates.pkl"), 'rb'))

        # Define the window size
        self.full_patch_size = 25
        self.center = self.full_patch_size // 2
        self.window_size = self.patch_size[0] // 2

        # Get the cat2vec LC embeddings
        if self.use_landcover and self.landcover_encoding == 'cat2vec':
            landcover_embeddings = pd.read_csv(
                join(self.cat2vec_path, "embeddings_train.csv")
            )
            landcover_embeddings = dict([
                (v, np.array([a, b, c, d, e]))
                for v, a, b, c, d, e in
                zip(
                    landcover_embeddings.mapping,
                    landcover_embeddings.dim0,
                    landcover_embeddings.dim1,
                    landcover_embeddings.dim2,
                    landcover_embeddings.dim3,
                    landcover_embeddings.dim4
                )
            ])
            self.landcover_embeddings = landcover_embeddings

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):

        # Find the file, tile, and row index corresponding to this chunk
        file_name, tile_name, idx_chunk = find_index_for_chunk(
            self.index,
            idx,
            self.length
        )

        # Get the file handle
        f = self.handles[file_name]

        # Initialize the pixel slicing bounds
        start = self.center - self.window_size
        end = self.center + self.window_size + 1

        # Initialize the data list
        data = []

        # Latitude and longitude data
        lat_offset = f[tile_name]['GEDI']['lat_offset'][idx_chunk]
        lat_decimal = f[tile_name]['GEDI']['lat_decimal'][idx_chunk]
        lon_offset = f[tile_name]['GEDI']['lon_offset'][idx_chunk]
        lon_decimal = f[tile_name]['GEDI']['lon_decimal'][idx_chunk]
        lat = np.sign(lat_decimal) * (np.abs(lat_decimal) + lat_offset)
        lon = np.sign(lon_decimal) * (np.abs(lon_decimal) + lon_offset)
        lat_cos, lat_sin, lon_cos, lon_sin = encode_coords(
            lat,
            lon,
            self.patch_size
        )
        if self.use_latlon:
            data.extend([
                lat_cos[..., np.newaxis],
                lat_sin[..., np.newaxis],
                lon_cos[..., np.newaxis],
                lon_sin[..., np.newaxis]
            ])
        else:
            data.extend([
                lat_cos[..., np.newaxis],
                lat_sin[..., np.newaxis]
            ])

        # GEDI dates
        gedi_num_days = f[tile_name]['GEDI']['date'][idx_chunk]

        if self.use_gedi_dates:
            gedi_doy_cos, gedi_doy_sin, gedi_date = get_doy(
                gedi_num_days,
                self.patch_size
            )
            gedi_num_days = np.full(
                (self.patch_size[0], self.patch_size[1]),
                gedi_num_days
            ).astype(np.float32)
            gedi_num_days = normalize_data(
                gedi_num_days,
                self.norm_values['GEDI']['date'],
                'min_max' if self.normalization_strategy == 'pct' else self.normalization_strategy
            )
            data.extend([
                gedi_num_days[..., np.newaxis],
                gedi_doy_cos[..., np.newaxis],
                gedi_doy_sin[..., np.newaxis]
            ])

        # Sentinel-2 bands
        if isinstance(self.use_s2_bands, list) and self.use_s2_bands != []:
            # Set the order and indices for the Sentinel-2 bands
            if not hasattr(self, 's2_order'):
                self.s2_order = list(f[tile_name]['S2_bands'].attrs['order'])
            if not hasattr(self, 's2_indices'):
                self.s2_indices = [
                    self.s2_order.index(band)
                    for band in self.use_s2_bands
                ]

            # Get the bands
            s2_bands = f[tile_name]['S2_bands'][idx_chunk, start:end, start:end, :]
            s2_bands = s2_bands.astype(np.float32)

            # Get the BOA offset, if it exists
            if 'S2_boa_offset' in f[tile_name]['Sentinel_metadata'].keys():
                s2_boa_offset = f[tile_name]['Sentinel_metadata']['S2_boa_offset'][idx_chunk]
            else:
                s2_boa_offset = 0

            # Get the surface reflectance values
            sr_bands = (s2_bands - s2_boa_offset.astype('float32') * 1000) / 10000
            sr_bands[s2_bands == 0] = 0
            sr_bands[sr_bands < 0] = 0
            s2_bands = sr_bands

            # Normalize the bands
            s2_bands = normalize_bands(
                s2_bands,
                self.norm_values['S2_bands'],
                self.s2_order,
                self.normalization_strategy,
                NODATAVALS['S2_bands']
            )
            s2_bands = s2_bands[:, :, self.s2_indices]
            data.extend([s2_bands])

            # Sentinel-2 date
            if self.use_s2_day:
                s2_num_days = f[tile_name]['Sentinel_metadata']['S2_date'][idx_chunk]
                s2_num_days = np.full(
                    (self.patch_size[0], self.patch_size[1]),
                    s2_num_days
                ).astype(np.float32)
                s2_num_days = normalize_data(
                    s2_num_days,
                    self.norm_values['Sentinel_metadata']['S2_date'],
                    'min_max' if self.normalization_strategy == 'pct' else self.normalization_strategy
                )
                data.extend([s2_num_days[..., np.newaxis]])
            if self.use_s2_doy:
                s2_num_days = f[tile_name]['Sentinel_metadata']['S2_date'][idx_chunk]
                s2_doy_cos, s2_doy_sin, s2_date = get_doy(s2_num_days, self.patch_size)
                data.extend([s2_doy_cos[..., np.newaxis], s2_doy_sin[..., np.newaxis]])
            if self.use_s2_sun:
                azimuth_cos = f[tile_name]['Sentinel_metadata']['S2_sun_azimuth_cos'][idx_chunk]
                azimuth_sin = f[tile_name]['Sentinel_metadata']['S2_sun_azimuth_sin'][idx_chunk]
                altitude = f[tile_name]['Sentinel_metadata']['S2_sun_altitude'][idx_chunk]
                azimuth_cos_patch = np.full(
                    (self.patch_size[0], self.patch_size[1]),
                    azimuth_cos
                ).astype(np.float32)
                azimuth_sin_patch = np.full(
                    (self.patch_size[0], self.patch_size[1]),
                    azimuth_sin
                ).astype(np.float32)
                altitude_patch = np.full(
                    (self.patch_size[0], self.patch_size[1]),
                    altitude
                ).astype(np.float32)
                data.extend([
                    azimuth_cos_patch[..., np.newaxis],
                    azimuth_sin_patch[..., np.newaxis],
                    altitude_patch[..., np.newaxis]
                ])

        # Sentinel-1 bands
        if self.use_s1:
            # Set the order for the Sentinel-1 bands
            if not hasattr(self, 's1_order'):
                self.s1_order = f[tile_name]['S1_bands'].attrs['order']

            s1_bands = f[tile_name]['S1_bands'][idx_chunk, start:end, start:end, :]
            s1_bands = s1_bands.astype(np.float32)
            s1_bands = normalize_bands(
                s1_bands,
                self.norm_values['S1_bands'],
                self.s1_order,
                self.normalization_strategy
            )

            s1_num_days = f[tile_name]['Sentinel_metadata']['S1_date'][idx_chunk, :]
            s1_doy_cos, s1_doy_sin, s1_date = get_doy(
                s1_num_days,
                self.patch_size
            )
            s1_num_days = np.full(
                (self.patch_size[0], self.patch_size[1]),
                s1_num_days
            ).astype(np.float32)
            s1_num_days = normalize_data(
                s1_num_days,
                self.norm_values['Sentinel_metadata']['S1_date'],
                'min_max' if self.normalization_strategy == 'pct' else self.normalization_strategy
            )
            data.extend([
                s1_bands,
                s1_num_days[..., np.newaxis],
                s1_doy_cos[..., np.newaxis],
                s1_doy_sin[..., np.newaxis]
            ])

        # ALOS bands
        if self.use_alos:
            # Set the order for the ALOS bands
            if not hasattr(self, 'alos_order'):
                self.alos_order = f[tile_name]['ALOS_bands'].attrs['order']

            # Get the bands
            alos_bands = f[tile_name]['ALOS_bands'][idx_chunk, start:end, start:end, :]
            alos_bands = alos_bands.astype(np.float32)

            # Get the gamma naught values
            alos_bands = np.where(
                alos_bands == NODATAVALS['ALOS_bands'],
                -9999.0,
                10 * np.log10(np.power(alos_bands.astype(np.float32), 2)) - 83.0
            )

            # Normalize the bands
            alos_bands = normalize_bands(
                alos_bands,
                self.norm_values['ALOS_bands'],
                self.alos_order,
                self.normalization_strategy,
                -9999.0
            )
            data.extend([alos_bands])

        # CH data
        if self.use_canopy_height:
            ch = f[tile_name]['CH']['ch'][idx_chunk, start:end, start:end]
            ch = normalize_data(
                ch,
                self.norm_values['CH']['ch'],
                self.normalization_strategy,
                NODATAVALS['CH']
            )
            ch_std = f[tile_name]['CH']['std'][
                idx_chunk, start:end, start:end]
            ch_std = normalize_data(
                ch_std,
                self.norm_values['CH']['std'],
                self.normalization_strategy,
                NODATAVALS['CH']
            )
            data.extend([ch[..., np.newaxis], ch_std[..., np.newaxis]])

        # LC data
        if self.use_landcover:
            lc = f[tile_name]['LC'][idx_chunk, start:end, start:end, :]

            if self.landcover_encoding == 'cat2vec':
                # cat2vec embeddings -> 5 dim
                lc, lc_prob = encode_landcover_cat2vec(
                    lc,
                    self.landcover_embeddings
                )
                data.extend([lc, lc_prob[..., np.newaxis]])

            elif self.landcover_encoding in ['onehot', 'distribution']:
                # one-hot encoding -> 14 dim
                lc_prob = np.where(
                    lc[:, :, 1] == NODATAVALS['LC'],
                    0,
                    lc[:, :, 1] / 100
                )
                if self.landcover_encoding == 'distribution':
                    lc = encode_landcover_distribution(lc[:, :, 0])
                else:
                    lc = encode_landcover_onehot(lc[:, :, 0])

                data.extend([lc, lc_prob[..., np.newaxis]])

            elif self.landcover_encoding == 'sincos':
                # sin/cosine encoding, 2dim
                lc_cos, lc_sin, lc_prob = encode_landcover(lc)
                data.extend([
                    lc_cos[..., np.newaxis],
                    lc_sin[..., np.newaxis],
                    lc_prob[..., np.newaxis]
                ])

            else:
                raise ValueError(
                    f"The landcover encoding strategy must be one of "
                    f"{self.LANDCOVER_ENCODINGS}"
                )

        # DEM data
        if self.use_dem or self.use_topo_aspect or self.use_topo_slope:
            dem = f[tile_name]['DEM'][idx_chunk, start:end, start:end]

            if self.use_topo_aspect or self.use_topo_slope:
                # Get the slope and aspect
                slope, aspect_cos, aspect_sin = get_topology(dem)
                if self.use_topo_slope:
                    data.extend([slope[..., np.newaxis]])
                if self.use_topo_aspect:
                    data.extend([
                        aspect_cos[..., np.newaxis],
                        aspect_sin[..., np.newaxis]
                    ])

            dem = normalize_data(
                dem,
                self.norm_values['DEM'],
                self.normalization_strategy,
                NODATAVALS['DEM']
            )
            data.extend([dem[..., np.newaxis]])

        # Concatenate the data together
        data = torch.from_numpy(
            np.concatenate(data, axis=-1).swapaxes(-1, 0)
        ).to(torch.float)

        # Get the GEDI target data
        agbd = f[tile_name]['GEDI']['agbd'][idx_chunk]
        if self.normalize_target:
            agbd = normalize_data(
                agbd,
                self.norm_values['GEDI']['agbd'],
                self.normalization_strategy
            )
        agbd = torch.from_numpy(np.array(agbd, dtype=np.float32)).to(torch.float)

        # Prepare the output format
        sample = {
            "image": data,
            "mask": agbd,
        }
        if self.use_metadata:
            sample["location_coords"] = torch.tensor([lat, lon], dtype=torch.float32)
            # TODO: in case date is needed, look into expected format
            #  sample["temporal_coords"] = ...

        return sample

    def _precompute_s2_sun_position(
            self,
            overwrite: bool = False,
            verbose: bool = True
    ):
        """Add the Sentinel-2 UTC acquisition time and corresponding sun
        position for each GEDI location to an already-existing AGBD HDF5
        file.

        Args:
        - overwrite (bool): Whether the UTC times and sun position
        information should be overwritten, if already found in the HDF5
        - verbose (bool): Verbosity
        """
        # Iterate over all files
        for fname in self.fnames:
            add_s2_sun_position_to_h5(
                join(self.h5_path, fname),
                self.s2_tile_to_dates,
                overwrite=overwrite,
                verbose=verbose
            )
        return


########################################################################
# AGBD utils
########################################################################

def initialize_index(
        fnames: list,
        mode: str,
        chunk_size: int,
        path_mapping: str,
        path_h5: str,
):
    """This function creates the index for the dataset. The index is a
    dictionary which maps the file names (`fnames`) to the tiles that
    are in the `mode` (train, val, test); and the tiles to the number
    of chunks that make it up.

    Args:
        fnames (list): list of file names
        mode (str): the mode of the dataset (train, val, test)
        chunk_size (int): the size of the chunks
        path_mapping (str): the path to the file mapping each mode to
            its tiles
        path_h5 (str): the path to the h5 file holding data

    Returns:
        idx (dict): dictionary mapping the file names to the tiles and
            the tiles to the chunks
        total_length (int): the total number of chunks in the dataset
    """

    # Load the mapping from mode to tile name
    with open(join(path_mapping, 'biomes_splits_to_name.pkl'), 'rb') as f:
        tile_mapping = pickle.load(f)

    # Iterate over all files
    idx = {}
    for fname in fnames:
        idx[fname] = {}

        with h5py.File(join(path_h5, fname), 'r') as f:

            # Get the tiles in this file which belong to the mode
            all_tiles = list(f.keys())
            tiles = np.intersect1d(all_tiles, tile_mapping[mode])

            # Iterate over the tiles
            for tile in tiles:
                # Get the number of patches in the tile
                n_patches = len(f[tile]['GEDI']['agbd'])
                idx[fname][tile] = n_patches // chunk_size

    total_length = sum(sum(v for v in d.values()) for d in idx.values())

    return idx, total_length


def find_index_for_chunk(
        index: dict,
        n: int,
        total_length: int,
):
    """For a given `index` and `n`-th chunk, find the file, tile, and
    row index corresponding to this chunk.

    Args:
        index (dict): dictionary mapping the files to the tiles and the
            tiles to the chunks
        n (int): the n-th chunk
        total_length (int):

    Returns:
        file_name (str): the name of the file
        tile_name (str): the name of the tile
        chunk_within_tile (int): the chunk index within the tile
    """
    # Check that the chunk index is within bounds
    if n >= total_length:
        raise IndexError(
            f"Index {n} out of bounds for dataset of length {total_length}"
        )

    # Iterate over the index to find the file, tile, and row index
    cumulative_sum = 0
    for file_name, file_data in index.items():
        for tile_name, num_rows in file_data.items():
            if cumulative_sum + num_rows > n:
                chunk_within_tile = n - cumulative_sum
                return file_name, tile_name, chunk_within_tile
            cumulative_sum += num_rows
    raise RuntimeError("Failed to locate chunk index – check index consistency")


def encode_lat_lon(
        lat: float,
        lon: float,
):
    """Encode the latitude and longitude into sin/cosine values. We use
    a simple WRAP positional encoding, as Mac Aodha et al. (2019).

    Args:
        lat (float): the latitude
        lon (float): the longitude

    Returns:
        (lat_cos, lat_sin, lon_cos, lon_sin) (tuple): the sin/cosine
            values for the latitude and longitude
    """

    # The latitude goes from -90 to 90
    lat_cos, lat_sin = np.cos(np.pi * lat / 90), np.sin(np.pi * lat / 90)
    # The longitude goes from -180 to 180
    lon_cos, lon_sin = np.cos(np.pi * lon / 180), np.sin(np.pi * lon / 180)

    # Now we put everything in the [0,1] range
    lat_cos, lat_sin = (lat_cos + 1) / 2, (lat_sin + 1) / 2
    lon_cos, lon_sin = (lon_cos + 1) / 2, (lon_sin + 1) / 2

    return lat_cos, lat_sin, lon_cos, lon_sin


def encode_coords(
        central_lat: float,
        central_lon: float,
        patch_size: tuple,
        resolution: int = 10
):
    """This function computes the latitude and longitude of a patch,
    from the latitude and longitude of its central pixel. It then
    encodes these values into sin/cosine values, and scales the results
    to [0,1].

    Args:
        central_lat (float): the latitude of the central pixel
        central_lon (float): the longitude of the central pixel
        patch_size (tuple): the size of the patch
        resolution (int): the resolution of the patch

    Returns:
        (lat_cos, lat_sin, lon_cos, lon_sin) (tuple): the sin/cosine
            values for the latitude and longitude
    """

    # Initialize arrays to store latitude and longitude coordinates

    i_indices, j_indices = np.indices(patch_size)

    # Calculate the distance offset in meters for each pixel
    offset_lat = (i_indices - patch_size[0] // 2) * resolution
    offset_lon = (j_indices - patch_size[1] // 2) * resolution

    # Calculate the latitude and longitude for each pixel
    # cf. https://stackoverflow.com/questions/7477003/calculating-new-longitude-latitude-from-old-n-meters
    # the volumetric mean radius of the Earth is 6371km, cf. https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
    latitudes = central_lat + (offset_lat / 6371000) * (180 / np.pi)
    longitudes = central_lon + (offset_lon / 6371000) * (180 / np.pi) / np.cos(latitudes * np.pi / 180)

    # Encode the latitude and longitude
    lat_cos, lat_sin, lon_cos, lon_sin = encode_lat_lon(latitudes, longitudes)

    return lat_cos, lat_sin, lon_cos, lon_sin


def get_doy(
        num_days: int,
        patch_size: tuple[int, int],
):
    """For a given number of days before/since the start of the GEDI
    mission, this function calculates the day of year (number between 1
    and 365) and encodes it into sin/cosine values.

    Args:
        num_days (int): the number of days before/since the start of the
            GEDI mission
        patch_size (tuple[int, int]): width and height of the output
            patch

    Returns:
        (doy_cos, doy_sin, date) (tuple): the sin/cosine values for the
            day of year as well as the absolute datetime date
    """

    # Get the date of acquisition and day of year
    start_date = datetime.strptime(GEDI_START_MISSION, '%Y-%m-%d')
    target_date = start_date + timedelta(days=int(num_days))
    doy = target_date.timetuple().tm_yday - 1  # range [1, 366]

    # Get the doy_cos and doy_sin
    doy_cos = np.cos(2 * np.pi * doy / 365)
    doy_sin = np.sin(2 * np.pi * doy / 365)

    # Now we put everything in the [0,1] range
    doy_cos, doy_sin = (doy_cos + 1) / 2, (doy_sin + 1) / 2

    return (
        np.full((patch_size[0], patch_size[1]), doy_cos),
        np.full((patch_size[0], patch_size[1]), doy_sin),
        target_date)


def get_sun_position(
        lat: float,
        lon: float,
        dates: str,
        utc_times: str,
        encode_azimuth: bool = True,
        encode_altitude: bool = True):
    """
    Compute Sun azimuth and elevation for vectorized input.

    Args:
        lat (array): Array of latitudes.
        lon (array): Array of longitudes.
        dates (array): Array of date strings (YYYY-MM-DD).
        utc_times (array): Array of UTC time strings (HH:MM:SS).

    Returns:
        tuple: (azimuths, altitudes) in degrees.
    """
    # Combine date and time into ISO format (YYYY-MM-DDTHH:MM:SS)
    datetime_strings = np.char.add(dates, "T")
    datetime_strings = np.char.add(datetime_strings, utc_times)

    # Convert to astropy Time object
    time = Time(datetime_strings, scale="utc")

    # Define observer location (broadcasting ensures shape match)
    location = EarthLocation(lat=lat * units.deg, lon=lon * units.deg)

    # Define AltAz frame
    altaz = AltAz(obstime=time, location=location)

    # Compute the Sun's position
    sun = get_sun(time).transform_to(altaz)

    azimuth = (np.cos(sun.az.rad), np.sin(sun.az.rad)) if encode_azimuth \
        else sun.az.deg
    altitude = sun.alt.deg / 90 if encode_altitude else sun.alt.deg

    return azimuth, altitude


def add_s2_sun_position_to_h5(
        h5_file_path: str,
        s2_tile_to_dates: dict,
        overwrite: bool = False,
        verbose: bool = True):
    """Add the Sentinel-2 UTC acquisition time and corresponding sun
    position for each GEDI location to an already-existing AGBD HDF5
    file.

    Args:
        h5_file_path (str): Path to the AGBD HDF5 file to extend
        s2_tile_to_dates (dict): Dictionary mapping S2 tiles and
            acquisition dates to the corresponding UTC times
        overwrite (bool): Whether the UTC times and sun position
            information should be overwritten, if already found in the
            HDF5
        verbose (bool): Verbosity
    """
    with (h5py.File(h5_file_path, 'r+') as f):
        enum = f.keys()
        if verbose:
            print(f"Processing S2 sun positions for: {h5_file_path}")
            enum = tqdm(f.keys())
        for tile_name in enum:

            # Some of the HDF5 files have tiles with no data. Skip those
            if len(f[tile_name]['GEDI']['lat_offset']) == 0:
                continue

            # Latitude and longitude data
            lat_offset = f[tile_name]['GEDI']['lat_offset'][:]
            lat_decimal = f[tile_name]['GEDI']['lat_decimal'][:]
            lon_offset = f[tile_name]['GEDI']['lon_offset'][:]
            lon_decimal = f[tile_name]['GEDI']['lon_decimal'][:]
            lat = np.sign(lat_decimal) * (np.abs(lat_decimal) + lat_offset)
            lon = np.sign(lon_decimal) * (np.abs(lon_decimal) + lon_offset)

            # Compute absolute S2 dates in parallel
            s2_num_days = f[tile_name]['Sentinel_metadata']['S2_date'][:]
            gedi_start = np.datetime64(GEDI_START_MISSION)
            s2_dates = gedi_start + s2_num_days.astype('timedelta64[D]')
            s2_dates_str = np.datetime_as_string(s2_dates, unit='D').astype('U10')
            s2_dates_str = np.char.replace(s2_dates_str, '-', '')

            # Vectorized lookup for UTC times
            func = lambda date: datetime.strptime(
                s2_tile_to_dates[tile_name][date], "%H%M%S").strftime("%H:%M:%S")
            utc_times = np.vectorize(func)(s2_dates_str)

            # Convert S2 dates format for get_sun_position
            func = lambda date: datetime.strptime(date, "%Y%m%d").strftime(
                "%Y-%m-%d")
            s2_dates = np.vectorize(func)(s2_dates_str)

            # Compute the sun position
            azimuth, altitude = get_sun_position(
                lat,
                lon,
                s2_dates,
                utc_times,
                encode_azimuth=True,
                encode_altitude=True)

            # Ensure the Sentinel_metadata group exists
            s2_metadata = f[tile_name]['Sentinel_metadata']
            extra = {
                'S2_utc_time': np.array(utc_times, dtype="S"),  # ASCII (bytes)
                'S2_sun_azimuth_cos': azimuth[0],
                'S2_sun_azimuth_sin': azimuth[1],
                'S2_sun_altitude': altitude}

            # Save results as new datasets in the HDF5 file
            for k, v in extra.items():
                if k in s2_metadata:
                    error_msg = (
                        f"['{tile_name}']['Sentinel_metadata']['{k}'] already "
                        f"exists in {h5_file_path}. If you want to overwrite "
                        f"with new values, please set overwrite=True.")
                    if overwrite:
                        del s2_metadata[k]
                    else:
                        raise ValueError(error_msg)
                s2_metadata.create_dataset(k, data=v)
    return


def get_topology(dem: np.ndarray):
    """This function computes the slope and aspect of the DEM.

    Resources:
    . https://www.spatialanalysisonline.com/HTML/gradient__slope_and_aspect.htm
    . https://gis.stackexchange.com/questions/361837/calculating-slope-of-numpy-array-using-gdal-demprocessing
    . https://math.stackexchange.com/a/3923660

    Args:
        dem (np.ndarray, shape batch_size, patch_size, patch_size): the
            Digital Elevation Model

    Returns:
        slope (np.ndarray): the slope of the DEM
        aspect_cos (np.ndarray): the cosine of the aspect of the DEM
        aspect_sin (np.ndarray): the sine of the aspect of the DEM
    """

    # Get the partial derivatives
    px, py = np.gradient(dem, 10, )
    # Get the slope, in [0,1]
    slope = np.sqrt(px ** 2 + py ** 2)
    # Get the aspect, in [0,2pi]
    aspect = np.pi / 2 - np.arctan2(py, px)
    aspect = np.where(aspect < 0, aspect + 2 * np.pi, aspect)
    # Encode and scale the aspect, in [0,1]
    aspect_cos = (np.cos(aspect) + 1) / 2
    aspect_sin = (np.sin(aspect) + 1) / 2

    return slope, aspect_cos, aspect_sin


def normalize_data(
        data: np.ndarray,
        norm_values: dict,
        normalization_strategy: str,
        nodata_value: float | int | None = None,
):
    """Normalize the data, according to various strategies:
    - mean_std: subtract the mean and divide by the standard deviation
    - pct: subtract the 1st percentile and divide by the 99th percentile
    - min_max: subtract the minimum and divide by the maximum

    Args:
        data (np.ndarray): the data to normalize
        norm_values (dict): the normalization values
        normalization_strategy (str): the normalization strategy

    Returns:
        normalized_data (np.ndarray): the normalized data
    """

    if normalization_strategy == 'mean_std':
        mean, std = norm_values['mean'], norm_values['std']
        if nodata_value is not None:
            data = np.where(data == nodata_value, 0, (data - mean) / std)
        else:
            data = (data - mean) / std

    elif normalization_strategy == 'pct':
        p1, p99 = norm_values['p1'], norm_values['p99']
        if nodata_value is not None:
            data = np.where(data == nodata_value, 0, (data - p1) / (p99 - p1))
        else:
            data = (data - p1) / (p99 - p1)
        data = np.clip(data, 0, 1)

    elif normalization_strategy == 'min_max':
        min_val, max_val = norm_values['min'], norm_values['max']
        if nodata_value is not None:
            data = np.where(
                data == nodata_value,
                0,
                (data - min_val) / (max_val - min_val))
        else:
            data = (data - min_val) / (max_val - min_val)

    else:
        raise ValueError(f'Normalization strategy `{normalization_strategy}` is not valid.')

    return data


def normalize_bands(
        bands_data: np.ndarray,
        norm_values: dict,
        order: list,
        normalization_strategy: str,
        nodata_value: int | float = None,
):
    """This function normalizes the bands data using the normalization
    values and strategy.

    Args:
        bands_data (np.ndarray): the bands data to normalize
        norm_values (dict): the normalization values
        order (list): the order of the bands
        normalization_strategy (str): the normalization strategy
        nodata_value (int/float): the nodata value

    Returns:
        bands_data (np.ndarray): the normalized bands data
    """

    for i, band in enumerate(order):
        band_norm = norm_values[band]
        bands_data[:, :, i] = normalize_data(
            bands_data[:, :, i],
            band_norm,
            normalization_strategy,
            nodata_value)

    return bands_data


def encode_landcover(lc_data: np.ndarray):
    """Encode the land cover classes into sin/cosine values and scale
    the class probabilities to [0,1].

    Args:
        lc_data (np.ndarray): the land cover data

    Returns:
        lc_cos (np.ndarray): the cosine values of the land cover classes
        lc_sin (np.ndarray): the sine values of the land cover classes
        lc_prob (np.ndarray): the land cover class probabilities
    """

    # Get the land cover classes
    lc_map = lc_data[:, :, 0]

    # Encode the LC classes with sin/cosine values and scale the data to [0,1]
    lc_cos = np.where(
        lc_map == NODATAVALS['LC'],
        0,
        (np.cos(2 * np.pi * lc_map / 100) + 1) / 2
    )
    lc_sin = np.where(
        lc_map == NODATAVALS['LC'],
        0,
        (np.sin(2 * np.pi * lc_map / 100) + 1) / 2
    )

    # Scale the class probabilities to [0,1]
    lc_prob = lc_data[:, :, 1]
    lc_prob = np.where(lc_prob == NODATAVALS['LC'], 0, lc_prob / 100)

    return lc_cos, lc_sin, lc_prob


def encode_landcover_cat2vec(lc_data: np.ndarray, embeddings: dict):
    """Embed the land cover classes using the cat2vec embeddings.

    Args:
        lc_data (np.ndarray): the land cover data
        embeddings (dict): the cat2vec embeddings

    Returns:
        lc_map (np.ndarray): the embedded land cover classes
        lc_prob (np.ndarray): the land cover class probabilities
    """

    # Get the land cover classes
    lc_map = lc_data[:, :, 0]
    lc_map = np.vectorize(
        lambda x: embeddings.get(x, embeddings.get(0)), signature='()->(n)')(lc_map)

    # Scale the class probabilities to [0,1]
    lc_prob = lc_data[:, :, 1]
    lc_prob = np.where(lc_prob == NODATAVALS['LC'], 0, lc_prob / 100)

    return lc_map, lc_prob


def encode_landcover_onehot(lc_data: np.ndarray):
    """Encode the land cover classes using one-hot encoding.

    Args:
        lc_data (np.ndarray): the land cover data

    Returns:
        lc_map (np.ndarray): the one-hot encoded land cover classes
    """
    # Number of classes
    num_classes = len(BIOMES_KEY2IDX)

    # Actually perform the one-hot encoding
    def one_hot(x):
        one_hot = np.zeros(num_classes)
        one_hot[BIOMES_KEY2IDX.get(x, 0)] = 1
        return one_hot

    one_hot_data = np.vectorize(
        one_hot, signature='() -> (n)')(lc_data).astype(np.float32)
    return one_hot_data


def encode_landcover_distribution(patch_lc: np.ndarray):
    """This function computes the distribution of biomes in a patch.

    Args:
        patch_lc (np.ndarray): the land cover classes in the patch, of
            size (patch_size, patch_size)

    Returns:
        biome_emb (np.ndarray): the biome distribution, of size
            (num_classes)
    """
    # Number of pixels in the patch
    num_pixels = patch_lc.size
    # Percentage of each biome in the patch
    counts = {
        value: np.count_nonzero(patch_lc == value) / num_pixels
        for value in BIOMES_KEYS
    }
    return np.array(list(counts.values())).astype(np.float32)
