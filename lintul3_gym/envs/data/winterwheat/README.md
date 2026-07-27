# Winter wheat (nitrogen-management use case)

LINTUL3 input files for the rain-fed winter-wheat nitrogen use case described in:

> Kallenberg, M. G. J., Overweg, H., van Bree, R., & Athanasiadis, I. N. (2023).
> Nitrogen management with reinforcement learning and crop growth models.
> *Environmental Data Science*, 2, e34. https://doi.org/10.1017/eds.2023.28

(see `examples/nitrogen-winterwheat/Paper-Cropgym.pdf`), copied from the paper's reference
implementation, [WUR-AI/PCSE-Gym](https://github.com/WUR-AI/PCSE-Gym/tree/master) (`master`
branch). This is the dataset the paper calibrated -- distinct from `lintul3_gym/envs/data/springwheat`, which was downloaded from [PCSE/LINTUL3 simulation example](https://pcse.readthedocs.io/en/stable/user_guide.html#running-a-simulation-with-pcse-lintul3).

## Files and their source

| File | Copied from (WUR-AI/PCSE-Gym) |
| --- | --- |
| `lintul3_winterwheat.crop` | `pcse_gym/envs/configs/crop/lintul3_winterwheat.crop` |
| `lintul3_winterwheat.site` | `pcse_gym/envs/configs/site/lintul3_springwheat.site` (upstream keeps the "springwheat" name, but these are the values used for the winter-wheat runs -- `WCI`/`WCSUBS` differ from `lintul3_gym/envs/data/springwheat`'s site file) |
| `lintul3_winterwheat.soil` | `pcse_gym/envs/configs/soil/lintul3_springwheat.soil` (identical values to `lintul3_gym/envs/data/springwheat`'s soil file) |
| `lintul3_winterwheat.agro` | `pcse_gym/envs/configs/agro/agromanagement_fertilization.yaml`, translated to this package's agro format (reference only -- see below) |
| `pcse-lintul3.patch` | `notebooks/nitrogen-winterwheat/pcse-lintul3.patch` -- the patch against `ajwdewit/pcse` this dataset's parameters were calibrated against; kept here for reference. See "Model-fidelity caveat" below. |

All three parameter files were downloaded verbatim (diffed byte-for-byte against the upstream
files) and only annotated with source-comment headers.

## Reproduction notebook

[`examples/nitrogen-winterwheat/PaperRep-Lintul3gym.ipynb`](../../../../examples/nitrogen-winterwheat/PaperRep-Lintul3gym.ipynb)
trains and evaluates against this dataset end-to-end (NASA POWER weather, the paper's train/test
locations and years, a `StandardPracticePolicy` baseline from `lintul3_gym.policies`, and a
Figure-4-style nitrogen-vs-rainfall plot). It supersedes the paper's own reference notebook,
`Paper-Cropgym.ipynb` (removed from this repo), which depended on the patched `pcse`/`gym`/SB3
stack from the paper's `v1.0.0` release rather than this package.

## Usage

`Lintul3Env` auto-discovers the `*.crop` / `*.site` / `*.soil` files in `data_dir` by extension, so
pointing it at this folder is enough. There is no bundled `nl1.xlsx`, so a NASA POWER
`WeatherConfig` is required -- pass one explicitly, as the tutorial notebooks do for their NASA
experiments. The crop calendar is built in code from a `SeasonConfig`, matching
`lintul3_winterwheat.agro`:

```python
from datetime import date
from pathlib import Path
from lintul3_gym import Lintul3Env, SeasonConfig, WeatherConfig

season = SeasonConfig(
    campaign_start=date(2007, 1, 1), crop_start=date(2007, 1, 1),
    crop_end=date(2007, 9, 1), crop_name="winter-wheat",
    crop_start_type="emergence", crop_end_type="earliest", max_duration=365,
)
env = Lintul3Env(
    data_dir=Path("lintul3_gym/envs/data/winterwheat"),
    season=season,
    decision_interval=7,
    weather=WeatherConfig(
        source="nasa",
        locations=((52, 5.5), (51.5, 5), (52.5, 6.0)),  # paper's training locations (NL)
        years=tuple(y for y in range(1990, 2022) if y % 2 == 1),  # paper's training years
        random_weather_per_episode=True,
    ),
)
```

Paper defaults, for reference (see `pcse_gym/utils/defaults.py` in the source repo):

| Setting | Value |
| --- | --- |
| Train locations (°N, °E) | `(52, 5.5)`, `(51.5, 5)`, `(52.5, 6.0)` |
| Test locations | `(52, 5.5)`, `(48, 0)` (out-of-distribution: Southern/French climate) |
| Train years | odd years, 1990-2021 |
| Test years | even years, 1990-2020 (`n=16`) |
| Crop features | `DVS, TGROWTH, LAI, NUPTT, TRAN, TNSOIL, TRAIN, TRANRF, WSO` -- matches `lintul3_gym`'s `CROP_FEATURES` exactly |
| Weather features | `IRRAD, TMIN, RAIN` -- `lintul3_gym`'s `WEATHER_FEATURES` additionally exposes `TMAX` |
| Nitrogen cost (β) | `10.0` -- matches `RewardConfig.nitrogen_cost` default |
| Reward | growth relative to a same-episode zero-nitrogen baseline, minus `β·N` -- matches `RewardConfig.relative_to_zero_nitrogen=True` |
| Action space (paper) | discrete `{0, 20, 40} kg N/ha` weekly; `lintul3_gym` instead exposes a continuous `Box` in g/m² |
| Standard Practice (SP) baseline | fixed total nitrogen split across three real dates -- Feb 24, Mar 26, Apr 29 -- via `lintul3_gym.policies.StandardPracticePolicy` (default total 17.07 g N/m² = 170.7 kg N/ha, the paper's reported SP amount); the paper's own code instead applies that total as a single early dose for its published numbers (see the caveat below) |
| Ceres baseline | not implemented -- it optimizes a per-episode dose using the *entire* season's future weather, which doesn't fit this package's transparent single-pass `Lintul3Env.step()` loop |

## Model-fidelity caveat

The paper's exact numbers were produced with a **patched** `pcse` fork
(`ajwdewit/pcse@7daa80a` + [`pcse-lintul3.patch`](pcse-lintul3.patch), also mirrored at
[WUR-AI/PCSE-Gym](https://github.com/WUR-AI/PCSE-Gym/blob/master/notebooks/nitrogen-winterwheat/pcse-lintul3.patch)),
not the stock `pcse==6.0.13` this project depends on. Three crop parameters in
`lintul3_winterwheat.crop` are only honoured by that patched model:

- `TNSOILI` (7.7 g N/m²) -- sets the initial soil-nitrogen state. Stock `pcse` always starts
  `TNSOIL` at 0.0, so nitrogen availability during the first weeks of the simulated winter will
  differ (confirmed by inspecting the installed `pcse` package: `Lintul3.initialize` has no
  `TNSOIL` override, and `RNSOIL` uses only `RNMIN`, not `RTMIN`).
- `RTMIN` -- an explicit net-mineralization-rate parameter distinct from `RNMIN`; stock `pcse`
  only reads `RNMIN` (which is also set here, so mineralization is not zero -- just less than the
  patched model's calibration intended).
- `FRTRL` -- fraction of stem reserves translocated to the grains once `DVS > DVSNT`. Stock
  `pcse`'s `relativeGrowthRates` has no translocation term, so simulated yield (`WSO`) will be
  somewhat lower than in the paper.

`RNMIN`, and the `NRF`/`GRF` nitrogen-vs-water growth-reduction logic the patch introduced, are
both already present in stock `pcse==6.0.13` (upstreamed since), so those behave identically.
`WMTAB` in `lintul3_winterwheat.site` is likewise only meaningful to the patched
`Lintul3Soil` (it lets an agromanagement file drive irrigation); this use case applies no
irrigation, so it has no effect either way.

Net effect: this dataset reproduces the paper's calibrated parameters and driving data exactly,
and runs end-to-end against this package's `Lintul3Env`, but early-season nitrogen dynamics and
final yield will not exactly match the published figures without also porting the patch's crop
and soil physics.

## Third-party license

`lintul3_gym` as a whole is Apache-2.0 licensed (see the repository root `LICENSE`), but the files
in this directory (`lintul3_winterwheat.crop`, `.site`, `.soil`, `.agro`, and `pcse-lintul3.patch`)
are copied verbatim from [WUR-AI/PCSE-Gym](https://github.com/WUR-AI/PCSE-Gym), which is
**GPL-3.0-or-later** licensed. They remain under that license rather than Apache-2.0 -- see
`pcse-lintul3.patch`'s own header and the upstream repository's `LICENSE` for the full terms.
