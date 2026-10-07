# Data

paddyguard does not include images, metadata, weather data or model weights. Git ignores every file
in this folder except this README.

## Image sources

| Dataset | Content | Terms |
|---|---|---|
| Paddy Doctor (Kaggle competition `paddy-disease-classification`) | About 10,400 labelled paddy leaf images, 10 classes, with `variety` and `age` for each image | Kaggle competition rules. Cite the Paddy Doctor paper (Petchiammal et al., 2023) |
| Rice leaf disease collections on Kaggle or Mendeley Data | Folder-per-class photos | Read the licence of each collection |

Most public collections have no location and no date for each photo. Without them, the weather
branch is off and paddyguard trains the image-only model. paddyguard never uses one weather vector
for all images.

## Expected layout

```
data/paddy/
  metadata.csv
  images/<label>/<image_id>.jpg
```

`metadata.csv` columns:

| Column | Needed? | Meaning |
|---|---|---|
| `image_id` | Yes | Unique image name |
| `label` | Yes | Class name. The class list is the sorted set of labels |
| `path` | No | Path from the data folder. Default `images/<label>/<image_id>` |
| `field_id` | No | Field, farm or plant ID. Images of one field stay in one split |
| `latitude`, `longitude` | For weather | Location of the photo in decimal degrees |
| `date` | For weather | Date of the photo (`YYYY-MM-DD`) |
| `variety`, `age` | No | Extra metadata (not used by the models) |

For Paddy Doctor, rename `train.csv` to `metadata.csv` and set `path` to `train_images/<label>/<image_id>`.

## Weather source

| Item | Value |
|---|---|
| Provider | Open-Meteo historical weather archive, <https://open-meteo.com/en/docs/historical-weather-api> |
| Key | None. paddyguard reads no API key |
| Terms | Open-Meteo data are CC BY 4.0. Cite Open-Meteo. Respect the request limits of the free API |
| Variables | Daily mean, maximum and minimum temperature, mean relative humidity, precipitation sum, maximum wind speed |
| Cache | `weather_cache/` (git ignores it) |

## Synthetic data (no download)

`paddyguard synth --out data/synthetic` writes drawn leaf images and a `metadata.csv` with fields,
locations and dates. The class of each image depends on the synthetic weather before its date.
The images are drawings, not photos of real plants.
