"""
Script adapted from BTSBot project.
"""

from multiprocessing import Pool, cpu_count
from astropy.table import Table
from functools import partial
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import argparse
import io


def get_ps_image_table(ra, dec, filters="grizy"):
    """
    Query ps1filenames.py service to get a list of images

    ra, dec = position in degrees
    filters = string with filters to include. includes all by default
    Returns a table with the results

    Adapted from
    https://spacetelescope.github.io/mast_notebooks/notebooks/PanSTARRS/PS1_image/PS1_image.html
    """
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    # The final URL appends our query to the PS1 image service
    url = f"{service}?ra={ra}&dec={dec}&filters={filters}"
    # Read the ASCII table returned by the url
    table = Table.read(url, format='ascii')
    return table


def get_ps_url(ra, dec, size=252, im_format="jpeg", output_size=None):
    """
    Get URL for PanSTARRS images

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
    filters = string with filters to include. choose from "grizy"
    format = data format (options are "jpg", "png" or "fits")
    Returns a string with the URL

    Adapted from
    https://spacetelescope.github.io/mast_notebooks/notebooks/PanSTARRS/PS1_image/PS1_image.html
    """
    if output_size is None:
        output_size = size

    table = get_ps_image_table(ra, dec, filters="gri")
    url = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           f"ra={ra}&dec={dec}&size={size}&format={im_format}&output_size={output_size}")

    if not all(f in table['filter'] for f in ['g', 'r', 'i']):
        return None

    flist = ["irgzy".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    table = table[np.isin(table['filter'], ['g', 'r', 'i'])]

    for i, param in enumerate(["red", "green", "blue"]):
        url = url + f"&{param}={table['filename'][i]}"

    return url


def download_image_batch(batch, survey):
    """Download images for a batch of sources from the specified survey"""
    results = []
    for source in batch:
        try:
            if survey == 'LS':
                # Legacy Survey query
                url = "https://www.legacysurvey.org/viewer/jpeg-cutout?" + \
                    f"ra={source['ra']}&dec={source['dec']}&" + \
                      "size=63&layer=ls-dr10&pixscale=1&bands=griy"

                response = requests.get(url)
                image = Image.open(io.BytesIO(response.content))
                image_array = np.array(image, dtype=np.float16)

                empty_image = all(32 == image_array.flatten())
                results.append((source['objectId'], image_array, empty_image))

            elif survey == 'PS':
                # PanSTARRS query
                table = get_ps_image_table(source["ra"], source["dec"], filters="gri")
                file_by_filter = {
                    f: fname
                    for f, fname in zip(table["filter"], table["filename"])
                    if f in ("g", "r", "i")
                }

                filt_to_channel = {
                    'g': 'blue',
                    'r': 'green',
                    'i': 'red',
                }

                channels = []
                img_shape = None
                cutout_size = 252
                for filt in ("i", "r", "g"):  # i->red, r->green, g->blue
                    fname = file_by_filter.get(filt)
                    if fname is None:
                        channels.append(None)  # will be zeroed later
                        continue

                    url = (
                        "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
                        f"ra={source['ra']}&dec={source['dec']}&size={cutout_size}&format=jpeg"
                        f"&{filt_to_channel[filt]}={fname}"
                    )
                    r = requests.get(url, timeout=30)
                    r.raise_for_status()
                    band_img = Image.open(io.BytesIO(r.content)).convert("L")
                    arr = np.array(band_img).astype(np.float32)
                    img_shape = img_shape or arr.shape
                    channels.append(arr)
                rgb = np.zeros((3, cutout_size, cutout_size), dtype=np.float32)
                if img_shape is None:
                    results.append((source['objectId'], rgb, True))
                    continue
                
                for idx, band in enumerate(channels):
                    if band is not None:
                        band_max = band.max() if band.size else 1.0
                        if band_max > 0:
                            rgb[idx, :, :] = band / band_max
                results.append((source['objectId'], rgb, False))
            else:
                raise ValueError(f"Unknown survey: {survey}")

        except Exception as e:
            print(f"Error downloading image for objectId {source['objectId']}: {e}")
            results.append((source['objectId'], None, True))

    return results


def query_images(cand, survey, max_workers=None):
    """Query images for the given candidates from the specified survey"""
    img_cache = {}
    missing_col = f'missing_{survey.upper()}'  # "missing_PS" or "missing_LS"
    cand[missing_col] = False

    objs = cand[['objectId', 'ra', 'dec']].drop_duplicates('objectId')
    print(len(objs), "unique objects")

    if max_workers is None:
        max_workers = min(cpu_count(), len(objs))
    print("Using", max_workers, "workers")

    with Pool(processes=max_workers) as pool:
        # Split the dataframe into batches for parallel processing
        batch_size = max(1, len(objs) // (3 * max_workers))
        batches = [objs.iloc[i:i + batch_size] for i in range(0, len(objs), batch_size)]

        # Convert batches to list of dictionaries for easier processing
        batch_dicts = [batch.to_dict('records') for batch in batches]

        # Process batches in parallel
        results = list(tqdm(
            pool.imap(partial(download_image_batch, survey=survey), batch_dicts),
            total=len(batch_dicts),
            desc=f"Downloading {survey.upper()} image batches",
            unit="batch"
        ))

        # Flatten the list of lists into a single list
        results = [item for sublist in results for item in sublist]

        # Process results
        for object_id, image_array, is_empty in results:
            if image_array is not None:
                img_cache[object_id] = image_array

                if is_empty:
                    cand.loc[cand['objectId'] == object_id, missing_col] = True

    return cand, img_cache


def process_dataset(survey, split_to_process, dir, workers):
    """Process dataset for the specified survey, split, and version"""
    if split_to_process == "all":
        splits = ['train', 'val', 'test']
    else:
        splits = [split_to_process]

    for split in splits:
        print(f"Querying {survey.upper()} for {split} split...")

        # Load candidate data
        original_cand = pd.read_parquet(f"{dir}/{split}.parquet")
        cand = original_cand[['objectId', 'ra', 'dec']]
        cand['ra'] = [x[0] for x in cand['ra']]
        cand['dec'] = [x[0] for x in cand['dec']]
        cand['objectId'] = [x[0] for x in cand['objectId']]

        # Query images based on survey
        cand, img_cache = query_images(cand, survey, max_workers=workers)

        # Create image array
        imgs = []
        for i, idx in enumerate(cand.index):
            obj = cand.loc[idx]
            if obj[f'missing_{survey.upper()}'] == False:
                imgs.append(img_cache[obj['objectId']].astype(float).flatten())

        original_cand['ps'] = imgs
        original_cand.to_parquet(f"{dir}/{split}_{survey}.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Query color cutouts from PanSTARRS or Legacy Survey'
    )
    parser.add_argument(
        '--survey', type=str, required=True, choices=['PS', 'LS'],
        help='Survey to query from: PS (PanSTARRS) or LS (Legacy Survey)'
    )
    parser.add_argument(
        '--split', type=str, choices=['train', 'val', 'test', 'all'],
        help='Dataset split to process (default: train)'
    )
    parser.add_argument(
        '--dir', type=str, default='BTS_new',
        help='Version identifier for the dataset (default: BTS_new)'
    )
    parser.add_argument(
        '--workers', type=int, default=8,
        help='Number of workers to use for parallel processing (default: 8)'
    )

    args = parser.parse_args()

    process_dataset(
        survey=args.survey,
        split_to_process=args.split,
        dir=args.dir,
        workers=args.workers,
    )