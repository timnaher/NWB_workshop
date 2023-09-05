from dandi.dandiapi import DandiAPIClient
from fsspec.implementations.cached import CachingFileSystem
from fsspec import filesystem
from h5py import File
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
from scipy.signal import decimate


def load_nwbfile(dandiset_id,file_path):
    # add comment
    with DandiAPIClient() as client:
        asset  = client.get_dandiset(dandiset_id, 'draft').get_asset_by_path(file_path)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

    # Create a virtual filesystem based on the http protocol and use caching to save accessed data to RAM.
    fs          = filesystem("http")
    file_system = fs.open(s3_url, "rb")
    file        = File(file_system, mode="r")
    io          = NWBHDF5IO(file=file, load_namespaces=True)
    nwbfile     = io.read()
    return nwbfile