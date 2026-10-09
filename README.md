<div align="center">

# paddyguard — Weather-Aware Paddy Leaf Disease Detection with an Honest Ablation

**paddyguard is a paddy (rice) leaf disease classifier for researchers who want to test if weather helps image models. It takes leaf images and their field, location and date through these steps to a seeded image-versus-weather ablation:**

`validate` → `group fields and duplicates` → `lagged weather` → `train` → `evaluate once` → `compare variants`.

![Variants](https://img.shields.io/badge/Variants-image_%7C_weather_%7C_both-1F3864?style=for-the-badge)
![Augmentations](https://img.shields.io/badge/Augmentations-6_tested-2E5FD9?style=for-the-badge)
![CLI commands](https://img.shields.io/badge/CLI_commands-7-6E86E8?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-36_passing-3DA35B?style=for-the-badge)
![Offline demo](https://img.shields.io/badge/Offline_demo-Yes-F5C542?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-A0399B?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-optional-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ablation-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Open-Meteo](https://img.shields.io/badge/Open--Meteo-no_key-555555?style=flat-square)
![Pillow](https://img.shields.io/badge/Pillow-images-555555?style=flat-square)
![Docs](https://img.shields.io/badge/Docs-ASD--STE100-5D6D7E?style=flat-square)

**[Summary](#1-summary)** ·
**[Workflow](#4-the-end-to-end-workflow)** ·
**[Run it](#14-how-to-run-paddyguard)** ·
**[Configuration](#144-environment-variables)** ·
**[Known problems](#17-known-problems)** ·
**[Glossary](#19-glossary)**

</div>

> [!NOTE]
> This README uses ASD-STE100 Simplified Technical English. The writing rules and the project
> vocabulary are in [`docs/ste-style-guide.md`](docs/ste-style-guide.md). Each term in the
> [Glossary](#19-glossary) has only one meaning.

---

paddyguard classifies paddy leaf images into disease classes and tests one claim: does weather before the photo help the classifier?
Each image gets weather features from its own location in the 14 days before its date. If the metadata has no location and date, the weather branch is off. paddyguard never tiles one weather vector to all images.
Images of one field and near-duplicate shots stay in one split. Augmentation runs on the fly on training images only.
The ablation trains the image-only, weather-only and image + weather variants on the same splits and seeds, and compares them with a paired bootstrap.

This README is the **one location that explains all of paddyguard**. It gives these topics:

- the general design
- each component and its procedure, step by step
- the decision rules
- the data map
- the runbook
- the validation results and the known problems

| If you are… | Read |
|---|---|
| A manager or reviewer | [1](#1-summary), [3](#3-design-rules), [4](#4-the-end-to-end-workflow), [16](#16-validation-results), [18](#18-key-points) |
| A developer who joins the project | All sections, in sequence. Keep [14](#14-how-to-run-paddyguard) and [17](#17-known-problems) open while you work |
| A researcher who runs paddyguard | [14](#14-how-to-run-paddyguard), then the section for the component that you use |

---

## Table of contents

1. 🧭 [Summary](#1-summary)
2. 🏗️ [How paddyguard is built](#2-how-paddyguard-is-built)
   - 2.1 [Components](#21-components)
   - 2.2 [System context](#22-system-context)
   - 2.3 [Repository layout](#23-repository-layout)
3. 🛡️ [Design rules](#3-design-rules)
4. 🔄 [The end-to-end workflow](#4-the-end-to-end-workflow)
   - 4.1 [Full flow](#41-full-flow)
   - 4.2 [The life cycle of one image](#42-the-life-cycle-of-one-image)
   - 4.3 [Who does which step](#43-who-does-which-step)
5. 🔵 [Metadata and the class list](#5-metadata-and-the-class-list)
6. 🌦️ [Weather features](#6-weather-features)
7. 🟢 [Groups and splits](#7-groups-and-splits)
8. 🌫️ [Augmentations](#8-augmentations)
9. 🟣 [The CPU ablation](#9-the-cpu-ablation)
10. 🤖 [Torch models and two-phase fine-tuning](#10-torch-models-and-two-phase-fine-tuning)
11. 📏 [Metrics and comparisons](#11-metrics-and-comparisons)
12. ⚖️ [The decision rules](#12-the-decision-rules)
13. 🗂️ [Data and file map](#13-data-and-file-map)
14. ▶️ [How to run paddyguard](#14-how-to-run-paddyguard)
    - 14.1 [Prerequisites](#141-prerequisites) · 14.2 [Installation](#142-installation) · 14.3 [Run paddyguard](#143-run-paddyguard) · 14.4 [Environment variables](#144-environment-variables)
15. 🧩 [How to extend paddyguard](#15-how-to-extend-paddyguard)
16. ✅ [Validation results](#16-validation-results)
17. ⚠️ [Known problems](#17-known-problems)
18. 📌 [Key points](#18-key-points)
19. 📖 [Glossary](#19-glossary)
20. 📄 [License](#20-license)

---

## 1. Summary

**The problem.** A weather input can only help if it describes the place and the time of each photo. These questions are difficult:

- Where does the weather for each image come from, and how do you stop a constant weather vector?
- How do you keep augmented copies and near-duplicate shots out of validation and test?
- How do you keep one pixel scale from the file to the backbone?
- How do you show that weather adds value, with seeds and a confidence interval?
- How do you run the full path on a laptop with no GPU and no key?

paddyguard gives each of these questions its own component. Each component has a validated input and a tested output.

| Item | Value |
|---|---|
| Input | `metadata.csv` and the images. Location and date are optional |
| Output | Weather features, ablation JSON (three variants, seeds, CIs, paired comparison, robustness), torch runs with checkpoints |
| Components | **10** modules: config, metadata, weather, images, augment, splits, evaluate, baseline, deep, synthetic, plus the CLI |
| Weather providers | `offline` (synthetic climate) and `open-meteo` (historical archive, no key) |
| Backbones | `tiny_cnn` (CPU), `efficientnet_v2_s`, `convnext_tiny` (torchvision) |
| Offline mode | Synthetic images, offline weather, the CPU ablation and all tests. No download, no key, no GPU |
| Safety | No constant weather. Field and duplicate groups never cross a split. Augmentation on training images only |
| Tests | **36** pass in CI (`.[dev]` only). The torch module (6 tests) skips there. With the `torch` extra, all 42 pass |

```mermaid
flowchart LR
    IN["Images + metadata"] --> A["Validate"] --> B["Group fields and duplicates"] --> C["Split"]
    W["Daily weather at each location"] --> D["Lagged weather features"]
    C & D --> E["Train variants"] --> F["Evaluate on test"] --> OUT["Ablation and robustness report"]
```

---

## 2. How paddyguard is built

### 2.1 Components

| Component | Module | Purpose |
|---|---|---|
| Settings | `src/paddyguard/config.py` | Environment variables: folders, seed, image size, provider, cache, lag, device |
| Metadata | `src/paddyguard/metadata.py` | Schema checks, default paths, sorted class list, weather availability |
| Weather | `src/paddyguard/weather.py` | Providers, cache, lagged window features, constant check |
| Images | `src/paddyguard/images.py` | Streaming loader, image features, thumbnail signatures |
| Augmentations | `src/paddyguard/augment.py` | Six weather-style augmentations and `RandomWeather` |
| Splits | `src/paddyguard/splits.py` | Near-duplicate groups, split groups, seeded grouped splits |
| Metrics | `src/paddyguard/evaluate.py` | Macro-F1, recall, confusion matrix with names, ECE, bootstraps |
| CPU ablation | `src/paddyguard/baseline.py` | Image, weather and image + weather variants, robustness test |
| Torch models | `src/paddyguard/deep.py` | Backbones, `FusionNet` (none, late, FiLM), two-phase training |
| Synthetic data | `src/paddyguard/synthetic.py` | Drawn leaves, fields, dates and a weather-driven class mix |
| CLI | `src/paddyguard/cli.py` | `synth`, `validate`, `weather`, `ablation`, `train`, `augment-preview`, `demo` |

The component map shows which module calls which module. An arrow points from the caller to the module that it uses.

```mermaid
flowchart TB
    CLI["cli.py<br/>paddyguard command"]
    subgraph DATAIN["Data in"]
        CFG["config.py<br/>Settings.from_env"]
        META["metadata.py<br/>load_metadata, validate"]
        WX["weather.py<br/>make_provider, build_features"]
        SYN["synthetic.py<br/>generate"]
    end
    subgraph PREP["Images and splits"]
        IMG["images.py<br/>load_rgb, image_features"]
        AUG["augment.py<br/>AUGMENTATIONS, RandomWeather"]
        SPL["splits.py<br/>duplicate_groups, assign_splits"]
    end
    subgraph MODELS["Models and metrics"]
        BASE["baseline.py<br/>run_ablation"]
        DEEP["deep.py<br/>train_deep, extra torch"]
        EVA["evaluate.py<br/>metrics, bootstraps"]
    end

    CLI --> CFG
    CLI --> META
    CLI --> WX
    CLI --> SYN
    CLI --> SPL
    CLI --> BASE
    CLI --> DEEP
    CLI -- "augment-preview" --> AUG
    SYN --> WX
    SPL --> IMG
    BASE --> SPL
    BASE --> IMG
    BASE --> AUG
    BASE --> EVA
    DEEP --> IMG
    DEEP --> AUG
    DEEP --> EVA
```

### 2.2 System context

```mermaid
flowchart TB
    U["Researcher"] --> CLI["paddyguard CLI"]
    DS["Image dataset (manual download)"] --> D["data/ folder (git ignores it)"]
    SYN["Synthetic generator"] --> D
    D --> CLI
    OM["Open-Meteo archive (optional, no key)"] --> C["weather_cache/ (git ignores it)"]
    C --> CLI
    CLI --> R["runs/ folder: metrics, checkpoints"]
```

### 2.3 Repository layout

```
paddyguard/
├── .github/workflows/ci.yml     # CI: install .[dev], run pytest
├── data/README.md               # image and weather sources, layout, columns
├── docs/ste-style-guide.md      # writing rules and project vocabulary
├── src/paddyguard/
│   ├── config.py                # settings from environment variables
│   ├── metadata.py              # metadata schema and class list
│   ├── weather.py               # providers, cache, lagged features
│   ├── images.py                # loader, features, signatures
│   ├── augment.py               # weather-style augmentations
│   ├── splits.py                # duplicate groups and grouped splits
│   ├── evaluate.py              # metrics and bootstraps
│   ├── baseline.py              # CPU ablation and robustness test
│   ├── deep.py                  # torch models and training (optional)
│   ├── synthetic.py             # synthetic data
│   └── cli.py                   # command-line interface
├── tests/                       # 42 tests (6 need torch)
├── .env.example                 # variable names only, no key
└── pyproject.toml               # package, extras, console script
```

---

## 3. Design rules

### 3.1 Weather belongs to each image

`weather.build_features` gets the daily weather at the location of each image and takes the window before its date. `assert_informative` stops the run if a weather feature has one value for all images. If the metadata has no location and date, the weather branch is off.

### 3.2 No future weather

The window ends one day before the photo date. The weather of the photo day and of later days never enters the features.

### 3.3 Splits come before augmentation

`assign_splits` makes the test and validation splits from the original images. Augmentation runs inside the training data loader only. Thus no augmented copy of a training image can reach validation or test.

```mermaid
flowchart LR
    IMG[/"Original images<br/>and split groups"/] --> AS["assign_splits<br/>seeded, grouped"]
    AS --> TR["train"]
    AS --> VA["val"]
    AS --> TE["test"]
    TR --> RW["RandomWeather, p 0.5<br/>torch training loader only"]
    RW --> FIT["fit the model"]
    VA --> SEL["select C or the best epoch<br/>no augmentation"]
    TE --> EV[/"test metrics, once<br/>no augmentation"/]
    FIT --> SEL
    SEL --> EV
```

### 3.4 Fields and near-duplicates stay together

`split_groups` joins images that share a `field_id` or a near-duplicate link. `StratifiedGroupKFold` splits these groups. `assert_no_overlap` stops the run if a group crosses a split.

### 3.5 One pixel contract

`load_rgb` returns float32 RGB in [0, 1]. Each augmentation checks this contract and refuses other inputs. The torch dataset applies the ImageNet normalization once, after augmentation.

### 3.6 Same splits, same seeds, same model family

The ablation trains each variant on the same split for each seed, with the same model family. The test split measures each variant once. A paired bootstrap over split groups compares image + weather with image only.

### 3.7 No key, no secret

The Open-Meteo archive needs no key. paddyguard reads no API key, and `.env.example` contains no value.

---

## 4. The end-to-end workflow

### 4.1 Full flow

```mermaid
flowchart TD
    M[/"metadata.csv"/] --> V["metadata.validate"]
    I[/"images/"/] --> V
    V -- "problem" --> ERR[/"MetadataError, exit 1"/]
    V --> DUP["splits.duplicate_groups"]
    V --> WA{"location and date<br/>in every row?"}
    WA -->|"yes"| WF["weather.build_features (lagged window)"]
    WA -->|"no"| IO["image-only"]
    CACHE[("weather_cache/<br/>offline or open-meteo")] --> WF
    WF --> CHK{"assert_informative<br/>constant feature?"}
    CHK -- "yes" --> STOP[/"ValueError, exit 1"/]
    DUP --> G["split_groups (field + duplicate)"]
    G --> S["assign_splits (seeded)"]
    S --> AB["baseline.run_ablation"]
    S --> DP["deep.train_deep"]
    CHK -- "no" --> AB
    CHK -- "no" --> DP
    IO --> AB
    IO -- "fusion none only" --> DP
    AB --> R[/"summary, paired comparison, robustness<br/>ablation JSON"/]
    DP --> RR[("runs/<br/>metrics.json, best.pt")]
    R --> HUMAN{{"HUMAN<br/>researcher reads the seeds, CIs and p-value"}}
    RR --> HUMAN

    classDef human fill:#fff3cd,stroke:#b8901f,color:#3d2f00,font-weight:bold
    class HUMAN human
```

### 4.2 The life cycle of one image

```mermaid
stateDiagram-v2
    state "Row in metadata.csv" as Row
    state "Validated row" as Valid
    state "Has 8 weather features" as Weather
    state "Image only" as NoWeather
    state "In a split group" as Grouped
    state "In train" as Train
    state "In val" as Val
    state "In test" as Test
    state "Augmented in memory" as Aug
    state "Used to fit" as Fit
    state "Used to select C or the best epoch" as Select
    state "Predicted once" as Pred
    [*] --> Row
    Row --> MetadataError: column, range or file problem
    Row --> Valid: validate
    Valid --> Weather: build_features
    Valid --> NoWeather: no location or date
    Weather --> RunStopped: assert_informative finds a constant feature
    Weather --> Grouped: duplicate_groups and split_groups
    NoWeather --> Grouped: duplicate_groups and split_groups
    Grouped --> Train: assign_splits
    Grouped --> Val: assign_splits
    Grouped --> Test: assign_splits
    Train --> Aug: RandomWeather in the torch loader
    Train --> Fit: no augmentation
    Aug --> Fit
    Val --> Select
    Test --> Pred: each variant predicts its class
    Fit --> [*]
    Select --> [*]
    Pred --> [*]
    MetadataError --> [*]
    RunStopped --> [*]
```

1. `validate` checks its `image_id`, label, file, location and date.
2. `duplicate_groups` compares its thumbnail signature with the other images of its class.
3. `split_groups` gives it the group of its field and of its near-duplicates.
4. `build_features` gives it 8 weather features from the 14 days before its date.
5. The split puts the image, with its whole group, in `train`, `val` or `test`.
6. If the image is in `train`, the loader can change it with one random augmentation.
7. If the image is in `test`, each variant predicts its class once.

### 4.3 Who does which step

The sequence shows the `ablation` command on data with a location and a date for each image.

```mermaid
sequenceDiagram
    autonumber
    actor R as Researcher
    participant CLI as paddyguard CLI
    participant META as metadata.py
    participant WX as weather.py
    participant WC as weather_cache/
    participant OM as Open-Meteo archive
    participant SPL as splits.py
    participant BASE as baseline.py
    participant EVA as evaluate.py

    R->>CLI: paddyguard ablation --data data/paddy
    CLI->>CLI: Settings.from_env
    CLI->>META: load_metadata, then validate
    META-->>CLI: clean table
    CLI->>META: weather_available
    CLI->>WX: make_provider, build_features
    loop each location
        WX->>WC: read the CSV for the location and window
        alt not in the cache
            WX->>OM: GET daily weather (open-meteo provider)
            OM-->>WX: daily JSON
            WX->>WC: write the CSV
        end
    end
    WX-->>CLI: 8 features for each image
    CLI->>WX: assert_informative
    CLI->>SPL: duplicate_groups, then split_groups
    CLI->>BASE: run_ablation(meta, weather, groups, seeds)
    loop each seed
        BASE->>SPL: assign_splits(seed)
        BASE->>BASE: fit_select for each variant
        BASE->>EVA: metrics and group_bootstrap on test
    end
    BASE->>EVA: paired_bootstrap, first seed
    BASE-->>CLI: AblationResult
    CLI-->>R: summary lines and the ablation JSON
```

---

## 5. Metadata and the class list

**Purpose.** Make a clean metadata table and one fixed class list.

```mermaid
flowchart TD
    IN[/"data folder"/] --> EX{"metadata.csv exists?"}
    EX -- "no" --> FNF[/"FileNotFoundError"/]
    EX -- "yes" --> REQ{"image_id and label<br/>columns present?"}
    REQ -- "no" --> ERR[/"MetadataError<br/>list of problems"/]
    REQ -- "yes" --> UID["Check that image_id is unique"]
    UID --> DEF["Default path images/label/image_id<br/>default field_id = image_id"]
    DEF --> DT["Parse date<br/>count rows with no valid date"]
    DT --> RNG["Check latitude and longitude ranges"]
    RNG --> FILES["Check that each image file exists"]
    FILES --> ANY{"Any problem?"}
    ANY -- "yes" --> ERR
    ANY -- "no" --> OUT[/"Clean metadata table"/]
    OUT --> CLS["classes_of<br/>sorted set of labels"]
    OUT --> WAV["weather_available<br/>location and date in every row"]
```

| Input | Output |
|---|---|
| `metadata.csv` | Table with `image_id`, `label`, `path`, `field_id`, optional `latitude`, `longitude`, `date` |

**Procedure**

1. Check that `image_id` and `label` exist and that `image_id` is unique.
2. If `path` is missing, use `images/<label>/<image_id>`.
3. If `field_id` is missing, each image is its own field.
4. Parse `date` and check the ranges of `latitude` (−90 to 90) and `longitude` (−180 to 180).
5. Check that each image file exists.

The class list is the sorted set of labels. It never depends on the order of a folder listing.

---

## 6. Weather features

**Purpose.** Give each image the weather of its own place in the days before its photo.

```mermaid
flowchart TD
    IN[/"Metadata with location and date"/] --> GRP["Group the images by location<br/>rounded to 3 decimals"]
    GRP --> RANGE["Date range for the location<br/>first date minus lag to last date minus 1"]
    RANGE --> HIT{"CSV in weather_cache/?"}
    HIT -- "yes" --> READ["Read the cached daily rows"]
    HIT -- "no" --> PROV{"Provider"}
    PROV -- "offline" --> OFF["OfflineWeather<br/>synthetic climate"]
    PROV -- "open-meteo" --> OM["OpenMeteoArchive<br/>archive API, no key"]
    OFF --> WRITE[("weather_cache/<br/>one CSV for each request")]
    OM --> WRITE
    READ --> WIN["window_features for each image<br/>days date minus lag to date minus 1"]
    WRITE --> WIN
    WIN --> COV{"Half of the days<br/>or more present?"}
    COV -- "no" --> NAN["8 features missing"]
    COV -- "yes" --> F8["8 features"]
    NAN --> CHK{"assert_informative<br/>constant column?"}
    F8 --> CHK
    CHK -- "yes" --> STOP[/"ValueError, run stops"/]
    CHK -- "no" --> OUT[/"Weather feature table<br/>one row for each image"/]
```

| Input | Output |
|---|---|
| Metadata with location and date, a provider | 8 features for each image |

**Procedure**

1. Group the images by location (rounded to 3 decimals).
2. For each location, get the daily weather from the earliest window start to the latest window end.
3. For each image, take the days from date − 14 to date − 1.
4. Calculate the 8 features in the table below. If fewer than half of the days exist, the features are missing.
5. Check that no feature is constant over all images.

| Feature | Meaning |
|---|---|
| `w_tmean`, `w_tmax`, `w_tmin` | Mean of the daily mean, maximum and minimum temperature (°C) |
| `w_rh` | Mean daily relative humidity (%) |
| `w_precip_total` | Total precipitation (mm) |
| `w_rainy_days` | Days with 1 mm or more of precipitation |
| `w_humid_days` | Days with a mean relative humidity of 90% or more |
| `w_wind` | Mean of the daily maximum wind speed |

| Provider | Source | Network |
|---|---|---|
| `offline` | Deterministic synthetic climate from latitude, longitude and day | No |
| `open-meteo` | Open-Meteo historical archive API, standard library HTTP | Yes, cached in `weather_cache/` |

---

## 7. Groups and splits

**Purpose.** Make splits where no field and no near-duplicate shot crosses a split.

```mermaid
flowchart TD
    IN[/"Validated metadata and images"/] --> ND{"--no-dedup?"}
    ND -- "yes" --> FLD["Field groups only"]
    ND -- "no" --> SIG["thumbnail_signature<br/>16 x 16 gray, mean 0, SD 1"]
    SIG --> CMP["Compare each pair of the same class<br/>mean absolute difference"]
    CMP --> NEAR{"0.06 or less?"}
    NEAR -- "yes" --> JOIN["Join in one duplicate group<br/>union-find"]
    NEAR -- "no" --> KEEP["Keep separate"]
    JOIN --> MERGE["split_groups<br/>join duplicate groups with field_id"]
    KEEP --> MERGE
    FLD --> OUTER
    MERGE --> OUTER["StratifiedGroupKFold, 5 folds<br/>1 fold is test"]
    OUTER --> INNER["Second 5-fold split of the rest<br/>1 fold is val"]
    INNER --> CHK{"assert_no_overlap<br/>group in 2 splits?"}
    CHK -- "yes" --> LE[/"LeakageError"/]
    CHK -- "no" --> OUT[/"train, val or test for each image"/]
```

**Procedure**

1. Make a 16 × 16 gray thumbnail of each image, scaled to mean 0 and standard deviation 1.
2. Join two images of the same class if the mean absolute difference of their thumbnails is 0.06 or less.
3. Join each image group with its field.
4. Take one fold of a 5-fold `StratifiedGroupKFold` as `test` (about 20%).
5. Take one fold of a second 5-fold split of the rest as `val` (about 16% of all images).
6. Check that no group is in two splits.

The `validate` command prints the number of near-duplicate groups with 2 or more images.

---

## 8. Augmentations

`RandomWeather` applies the augmentations in the torch training loader. Each call does these steps:

```mermaid
flowchart LR
    IMG[/"Training image<br/>float32 RGB, 0 to 1"/] --> DRAW{"Random value<br/>less than p?"}
    DRAW -- "no" --> SAME[/"Image unchanged"/]
    DRAW -- "yes" --> PICK["Pick one name from AUGMENTATIONS<br/>fog, haze, rain, brightness, shadow, motion_blur"]
    PICK --> CHK{"_check<br/>shape, float32, range"}
    CHK -- "fail" --> ERR[/"ValueError or TypeError"/]
    CHK -- "pass" --> APPLY["Apply to a new array<br/>input stays unchanged"]
    APPLY --> OUT[/"Augmented image"/]
```

| Name | Change |
|---|---|
| `fog` | Blend with light gray, strength 0.2–0.5 |
| `haze` | Vertical gradient of light haze, stronger at the top |
| `rain` | Short light diagonal streaks, then 10% darker |
| `brightness` | Multiply by 0.7–1.3 |
| `shadow` | Darken one vertical band by a factor of 0.4–0.7 |
| `motion_blur` | Horizontal line kernel of 3–7 pixels |

`RandomWeather(p, seed)` applies one random augmentation with probability `p` (default 0.5 in training). Each function checks the pixel contract, returns a new array and leaves its input unchanged. The `augment-preview` command writes one image with each augmentation.

---

## 9. The CPU ablation

**Purpose.** Test if weather helps, with no deep learning framework.

```mermaid
flowchart TD
    IN[/"Metadata, images, weather, groups, seeds"/] --> IF["extract_image_features<br/>31 values for each image"]
    IF --> WQ{"Weather table?"}
    WQ -- "no" --> V1["Variant image only"]
    WQ -- "yes" --> V3["Variants image, weather,<br/>image+weather"]
    V1 --> SEED["For each seed<br/>assign_splits"]
    V3 --> SEED
    SEED --> FS["fit_select for each variant<br/>impute, scale, balanced logistic regression"]
    FS --> CG["Fit C = 0.1, 1, 10 on train<br/>select C by val macro-F1"]
    CG --> REFIT["Refit on train + val"]
    REFIT --> TEST["Predict test once<br/>metrics, group_bootstrap"]
    TEST --> FIRST{"First seed?"}
    FIRST -- "yes, weather present" --> PAIR["paired_bootstrap<br/>image+weather minus image"]
    FIRST -- "yes" --> ROB["Robustness: image model<br/>on each augmented test set"]
    TEST --> SUM["seed_summary<br/>mean and SD of macro-F1"]
    PAIR --> OUT[/"AblationResult<br/>ablation JSON with --out"/]
    ROB --> OUT
    SUM --> OUT
```

| Input | Output |
|---|---|
| Metadata, images, weather features, groups, seeds | Macro-F1 of each variant and seed with CIs, mean ± standard deviation, paired comparison, robustness table |

**Procedure**

1. Make 31 image features for each image: HSV histograms (3 × 8 bins), shares of green, brown, yellow and pale pixels, and gradient and contrast statistics.
2. For each seed, make the splits.
3. For each variant, fit a scaler and a balanced logistic regression with C = 0.1, 1 and 10 on `train`. Select C by macro-F1 on `val`.
4. Refit on `train` + `val` with the selected C, and evaluate once on `test`.
5. For the first seed, compare image + weather with image only by a paired bootstrap over split groups.
6. For the first seed, run the image model on the test images after each augmentation.

| Variant | Features |
|---|---|
| `image` | 31 image features |
| `weather` | 8 weather features |
| `image+weather` | 39 features |

---

## 10. Torch models and two-phase fine-tuning

**Purpose.** Train a backbone with or without weather fusion.

```mermaid
flowchart TD
    IN[/"Metadata, split, weather features"/] --> FQ{"fusion none?"}
    FQ -- "no" --> STD["standardize<br/>train mean and SD, fill with train mean"]
    FQ -- "yes" --> DS
    STD --> DS["PaddyDataset<br/>stream load_rgb, RandomWeather on train only,<br/>then ImageNet normalization"]
    DS --> NET["build_backbone and FusionNet<br/>none, late or film"]
    NET --> LOSS["Class-weighted cross-entropy"]
    LOSS --> PH{"Pretrained, not tiny_cnn,<br/>epoch less than head_epochs?"}
    PH -- "yes" --> HEAD["Frozen backbone, head only<br/>lr 1e-3"]
    PH -- "no" --> ALL["All layers<br/>lr 1e-4, or 1e-3 in one-phase mode"]
    HEAD --> VAL["Validation macro-F1"]
    ALL --> VAL
    VAL --> BEST{"Better than the best?"}
    BEST -- "yes" --> CK[("best.pt")]
    BEST -- "no" --> PAT{"3 epochs with no gain<br/>and backbone not frozen?"}
    CK --> MORE{"Epochs left?"}
    PAT -- "no" --> MORE
    MORE -- "yes" --> PH
    PAT -- "yes" --> LOAD["Load best.pt<br/>predict test once"]
    MORE -- "no" --> LOAD
    LOAD --> OUT[/"metrics.json, config.json,<br/>test_proba.npy"/]
```

| Fusion | Method |
|---|---|
| `none` | Image features → dropout → linear head |
| `late` | Concatenate image features with a 32-unit weather embedding |
| `film` | Weather gives a scale and a shift for each image feature: f × (1 + γ) + β |

**Procedure**

1. Standardize the weather features with the mean and standard deviation of `train`. Fill missing values with the training mean.
2. Stream the images. Apply `RandomWeather` to training images only, then the ImageNet normalization.
3. Use a class-weighted cross-entropy loss with weights from the training class counts.
4. For a pretrained backbone, train the head for `--head-epochs` (default 3) with a frozen backbone at learning rate `1e-3`.
5. Then train all layers at learning rate `1e-4`.
6. After each epoch, measure the validation macro-F1. Keep the best checkpoint. Stop after 3 epochs with no improvement.
7. Load the best checkpoint and evaluate once on `test`.

`tiny_cnn` trains from random weights in one phase. It is for tests and CPU checks, not for a benchmark.

---

## 11. Metrics and comparisons

`evaluate.py` gives the metrics of one test split, the group intervals and the paired comparison.

```mermaid
flowchart LR
    Y[/"Test labels and probabilities"/] --> M["metrics<br/>macro_f1, balanced_accuracy,<br/>recall, confusion_matrix, ece"]
    Y --> GB["group_bootstrap<br/>resample whole split groups"]
    GB --> CI[/"macro_f1_ci, 95%"/]
    TWO[/"Probabilities of two variants<br/>on the same test images"/] --> PB["paired_bootstrap<br/>same group draw for both"]
    PB --> D[/"delta, 95% CI, p-value"/]
    SEEDS[/"Macro-F1 of each seed"/] --> SS["seed_summary"]
    SS --> MS[/"mean and SD"/]
```

| Metric | Meaning |
|---|---|
| `macro_f1` | Mean F1 over the classes in the test split. The main metric |
| `balanced_accuracy` | Mean recall over the classes |
| `recall` | Recall of each class, by name |
| `confusion_matrix` | Matrix with its class names |
| `ece` | Top-label expected calibration error, 10 bins |
| `macro_f1_ci` | 95% bootstrap interval over split groups |
| Paired comparison | Macro-F1 of image + weather minus image only, with a 95% CI and a p-value from a paired group bootstrap |
| Seed summary | Mean and standard deviation of macro-F1 over the seeds |

---

## 12. The decision rules

The map shows the step where each fixed value applies.

```mermaid
flowchart LR
    subgraph WXR["weather.py"]
        W1["Window: 14 days<br/>before the photo date"] --> W2["Coverage: half of the days"]
        W2 --> W3["Rainy day: 1 mm or more<br/>humid day: 90% or more"]
    end
    subgraph SPR["splits.py"]
        S1["Near-duplicate: 0.06"] --> S2["Test: 1 of 5 folds<br/>val: 1 of 5 folds of the rest"]
    end
    subgraph CPU["baseline.py"]
        C1["C grid: 0.1, 1, 10"]
    end
    subgraph TOR["deep.py DeepConfig"]
        T1["augment_p 0.5"] --> T2["head_epochs 3, finetune_lr 1e-4"]
        T2 --> T3["patience 3"]
    end
    W3 --> C1
    W3 --> T1
    S2 --> C1
    S2 --> T1
```

| Value | Where | Number |
|---|---|---|
| Weather window | `PADDYGUARD_WEATHER_LAG_DAYS` | 14 days before the photo date |
| Rainy day | `weather.RAINY_MM` | 1 mm or more |
| Humid day | `weather.HUMID_RH` | 90% or more |
| Minimum window coverage | `window_features` | half of the days |
| Near-duplicate distance | `duplicate_groups(max_distance=...)` | 0.06 mean absolute difference |
| Test split | `assign_splits(test_folds=...)` | 1 of 5 folds |
| Validation split | `assign_splits(val_folds=...)` | 1 of 5 folds of the rest |
| C grid | `baseline.C_GRID` | 0.1, 1, 10 |
| Augmentation probability | `DeepConfig.augment_p` | 0.5 |
| Head epochs, fine-tune rate | `DeepConfig` | 3 epochs, `1e-4` |
| Early stopping | `DeepConfig.patience` | 3 epochs |

---

## 13. Data and file map

| Path | Committed? | Contents |
|---|---|---|
| `data/README.md` | Yes | Sources, layout, columns, weather terms |
| `data/paddy/` | No (git ignores `/data/*`) | Your images and `metadata.csv` |
| `data/synthetic/` | No | Output of `paddyguard synth` |
| `weather_cache/` | No | Cached daily weather (CSV) |
| `runs/weather_features.csv` | No | Output of `paddyguard weather` |
| `runs/<backbone>_<fusion>_seed<n>/` | No | `best.pt`, `config.json`, `metrics.json`, `test_proba.npy` |
| `outputs/augment_preview/` | No | Output of `augment-preview` |
| `.env` | No | Local settings |

---

## 14. How to run paddyguard

### 14.1 Prerequisites

| Need | For |
|---|---|
| Python 3.11+ | All components |
| A labelled paddy image dataset | Real results (see [`data/README.md`](data/README.md)) |
| Location and date for each image | The weather variants |
| Internet access | The `open-meteo` provider only |
| `torch`, `torchvision` (extra `torch`) | `train` and the torch tests |

### 14.2 Installation

```bash
git clone https://github.com/KrishnaAnnavaram/paddyguard.git
cd paddyguard
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"         # add ,torch for the deep models
```

### 14.3 Run paddyguard

Offline demo (synthetic data, CPU, about 20 seconds):

```bash
paddyguard demo
```

Step by step. The commands run in this sequence:

```mermaid
flowchart LR
    SYN["paddyguard synth"] --> D[("data/synthetic/<br/>metadata.csv, images/")]
    REAL[/"data/paddy/<br/>your images"/] --> VAL
    D --> VAL["validate"]
    VAL --> WX["weather<br/>optional"]
    VAL --> AB["ablation"]
    VAL --> TR["train<br/>torch extra"]
    WX --> F[("runs/weather_features.csv")]
    AB --> J[("runs/ablation.json")]
    TR --> R[("runs/backbone_fusion_seedN/")]
```

```bash
paddyguard synth --out data/synthetic --fields 30 --photos 25 --seed 0
paddyguard validate --data data/synthetic
paddyguard weather --data data/synthetic --provider offline
paddyguard ablation --data data/synthetic --seeds 0 1 2 --out runs/ablation.json
paddyguard train --data data/synthetic --backbone tiny_cnn --fusion late --epochs 8 --image-size 64 --no-pretrained --seeds 0 1
paddyguard augment-preview --image data/synthetic/images/blast/f000_000.png --size 128
```

On real data with locations and dates:

```bash
export PADDYGUARD_WEATHER_PROVIDER=open-meteo
paddyguard validate --data data/paddy
paddyguard ablation --data data/paddy
paddyguard train --data data/paddy --backbone efficientnet_v2_s --fusion none --seeds 0 1 2
paddyguard train --data data/paddy --backbone efficientnet_v2_s --fusion late --seeds 0 1 2
```

| Command | What it does |
|---|---|
| `synth` | Writes drawn leaf images and `metadata.csv` |
| `validate` | Checks metadata and files, counts near-duplicate groups, reports if the weather branch is available |
| `weather` | Writes the 8 weather features of each image and prints their statistics |
| `ablation` | Image vs. weather vs. image + weather on the same splits and seeds (CPU) |
| `train` | Trains a torch backbone with `none`, `late` or `film` fusion for each seed |
| `augment-preview` | Writes one image with each augmentation |
| `demo` | Synthetic data, validation and the CPU ablation |

Exit codes: 0 for success, 1 for an error (bad metadata, missing files, no weather keys for `weather` or a fusion model, constant weather).

### 14.4 Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `PADDYGUARD_DATA_DIR` | `config.Settings` | Default data folder (default `data/paddy`) |
| `PADDYGUARD_OUTPUT_DIR` | `weather`, `train` | Run folder (default `runs`) |
| `PADDYGUARD_SEED` | `augment-preview` | Seed (default 42). The ablation and training seeds come from `--seeds` |
| `PADDYGUARD_IMAGE_SIZE` | `train` | Input size (default 224, minimum 16) |
| `PADDYGUARD_WEATHER_PROVIDER` | weather features | `offline` (default) or `open-meteo` |
| `PADDYGUARD_WEATHER_CACHE` | weather features | Cache folder (default `weather_cache`) |
| `PADDYGUARD_WEATHER_LAG_DAYS` | weather features | Window length in days (default 14) |
| `PADDYGUARD_DEVICE` | `train` | `auto`, `cpu` or `cuda` (default `auto`) |

paddyguard needs no credentials. `.env.example` gives the variable names. Git ignores `.env`.

---

## 15. How to extend paddyguard

| You want to… | Do this | Code change? |
|---|---|---|
| Add a weather provider | Write a class with a `daily(latitude, longitude, start, end)` method and add it to `make_provider` | Small |
| Add a weather feature | Add the value in `window_features` and its name in `FEATURES` | Small |
| Add an augmentation | Write a function with the pixel contract and add it to `AUGMENTATIONS` | Small |
| Add a backbone | Add a branch to `deep.build_backbone` and its name to `BACKBONES` | Small |
| Change the window length | Set `PADDYGUARD_WEATHER_LAG_DAYS` | No |
| Add agronomic features (variety, crop age) | Add them as a fourth ablation variant in `baseline.run_ablation` | Yes |

---

## 16. Validation results

All numbers below come from this repository. The model numbers use **synthetic data** (`paddyguard demo`, 30 fields × 25 photos, 6 classes, seed 0). They do not describe real paddy fields.

| Validation | Result | Command |
|---|---|---|
| Unit tests | CI installs only `.[dev]`: **36 passed**, 1 skipped (the torch module, `torch` extra). With the extras: 42 passed | `pytest -q` |
| Metadata check (synthetic) | 750 images, 6 classes, 30 fields, 52 near-duplicate groups | `paddyguard validate` |

CPU ablation on synthetic data, 3 seeds:

| Variant | Macro-F1, mean ± SD over 3 seeds | First seed (95% group CI) |
|---|---|---|
| `image` | 0.935 ± 0.025 | 0.907 (0.860–0.949) |
| `weather` | 0.267 ± 0.008 | 0.266 (0.219–0.299) |
| `image+weather` | 0.933 ± 0.025 | 0.904 (0.849–0.961) |

| Comparison or test | Result |
|---|---|
| `image+weather` minus `image`, first seed | −0.002, 95% paired CI −0.014 to +0.032, p = 0.970 |
| Image model on augmented test images (macro-F1) | clean 0.907, fog 0.033, haze 0.065, rain 0.239, brightness 0.698, shadow 0.710, motion blur 0.282 |

`tiny_cnn` on the same synthetic data (CPU, 8 epochs, 64 px, augmentation on, seeds 0 and 1):

| Fusion | Test macro-F1, seed 0 | Test macro-F1, seed 1 |
|---|---|---|
| `none` | 0.680 | 0.599 |
| `late` | 0.737 | 0.753 |
| `film` | 0.738 | 0.766 |

With the small CNN, the image branch is weaker, and both fusion modes give a higher macro-F1 on both seeds. Two seeds are not enough for a significance claim.

The weather-only variant is above the chance level of 0.167, so the weather features carry information. The image features already separate the classes well, so the fusion adds no significant macro-F1 on this data.
The robustness test shows that the image model trained with no augmentation fails on fog and haze. This result supports augmentation in training.
These results prove that the ablation and the checks work. They do not answer the question for real fields.

The prototype reported an accuracy for EfficientNetV2-L with a weather branch. That number is a prototype result, not reproduced here. Its weather input was one constant vector for all images, so this project does not compare with it.

---

## 17. Known problems

Read these problems before you use paddyguard for a publication.

| # | Area | Problem | Impact and action |
|---|---|---|---|
| 1 | Real data | CI runs on synthetic data only. Results on real images are not reproduced in CI | Run `ablation` and `train` on your data and report the seeds |
| 2 | Metadata | Most public paddy datasets have no location and no date for each photo | Without them, paddyguard can only train the image-only model |
| 3 | Weather grid | Open-Meteo gives gridded reanalysis data, not a field station | Small fields in one grid cell get the same weather |
| 4 | Near-duplicates | The thumbnail rule finds re-saves and second shots. It can miss a shot from another angle | Give a real `field_id` or plant ID when you have one |
| 5 | Class balance | Field datasets are often imbalanced | Read the recall of each class, not only macro-F1 |
| 6 | Synthetic images | The drawn leaves are much simpler than photos | Do not read the synthetic macro-F1 as a real performance |
| 7 | CPU speed | `efficientnet_v2_s` and `convnext_tiny` at 224 px are slow on a CPU | Use a GPU. The CPU path is for tests and small checks |
| 8 | Torch tests | CI installs only `.[dev]`, so the 6 torch tests skip in CI | Run `pip install -e ".[dev,torch]" && pytest` before a release |
| 9 | Advice | paddyguard gives a class, not a treatment | An agronomist must confirm a diagnosis before any spray decision |

---

## 18. Key points

1. **Weather is real or absent.** Each image gets the weather of its own place and time, and a constant weather vector stops the run.
2. **No leak through augmentation.** Splits come first, and augmentation runs on training images only.
3. **Fields and near-duplicates stay together.** Grouped splits keep them on one side.
4. **The claim is tested, not assumed.** Three variants, the same splits and seeds, and a paired bootstrap answer the weather question.
5. **One pixel scale.** Float32 RGB in [0, 1] from the file to the normalization step.
6. **It runs offline.** Synthetic data, offline weather and the CPU ablation need no key and no GPU.

---

## 19. Glossary

| Term | Meaning |
|---|---|
| **class** | One disease or health label |
| **field** | The group of images from one farm, plot or plant |
| **near-duplicate** | An image with a thumbnail signature close to another image of the same class |
| **split group** | Images joined by a shared field or a near-duplicate link |
| **weather window** | The days before the photo date that give the weather features |
| **weather features** | The 8 values from the weather window |
| **provider** | The source of daily weather: `offline` or `open-meteo` |
| **augmentation** | A weather-style change of a training image |
| **pixel contract** | float32 RGB values in [0, 1] |
| **variant** | `image`, `weather` or `image+weather` |
| **ablation** | The comparison of the variants on the same splits and seeds |
| **fusion** | How a torch model uses weather: `none`, `late` or `film` |
| **FiLM** | Feature-wise linear modulation: a scale and a shift for each feature |
| **macro-F1** | Mean F1 over the classes in the test split |
| **paired bootstrap** | A resample of split groups that measures two models on the same draw |

---

## 20. License

[MIT](LICENSE) © 2026 Krishna Annavaram
