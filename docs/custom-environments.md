# Custom environments

`Lintul3Env` isn't hard-coded to spring wheat: any crop with a LINTUL3 parameter set works, via
two pieces of configuration.

## What a data directory needs

- Exactly one `*.crop`, one `*.site`, and one `*.soil` file -- any filename, discovered by
  extension (see `resolve_paths()` in `lintul3_gym/envs/environment.py`). These are plain
  `PCSEFileReader`-format text files: crop physiology parameters, site/initial conditions, and
  soil hydraulic properties.
- Optionally, an `nl1.xlsx` if you want `WeatherConfig(source="excel")`. NASA POWER weather
  (`source="nasa"`) needs no local weather file at all.
- Optionally, a reference `.agro` file for provenance/documentation -- it is never read as an
  environment input; use `SeasonConfig` instead (see below).

## The crop calendar: `SeasonConfig`

```python
from datetime import date
from lintul3_gym import Lintul3Env, SeasonConfig, WeatherConfig

season = SeasonConfig(
    campaign_start=date(2007, 1, 1), crop_start=date(2007, 1, 1),
    crop_end=date(2007, 9, 1), crop_name="winter-wheat",
    crop_start_type="emergence", crop_end_type="earliest", max_duration=365,
)
env = Lintul3Env(data_dir="lintul3_gym/envs/data/winterwheat", season=season, decision_interval=7)
```

`SeasonConfig` mirrors a PCSE `.agro` file's `CropCalendar` section (campaign start, crop
start/end dates and types, `max_duration`) but is built in Python, so the calendar can vary
without editing YAML -- and so nitrogen application stays entirely under the RL agent's `step()`
actions rather than a fixed `TimedEvents` schedule.

## Worked examples in this repo

- [`lintul3_gym/envs/data/springwheat/`](../lintul3_gym/envs/data/springwheat) -- the bundled default (Netherlands,
  2006); see `examples/nitrogen-springwheat/Tutorial-Lintul3gym.ipynb` for the full walkthrough.
- [`lintul3_gym/envs/data/winterwheat/`](../lintul3_gym/envs/data/winterwheat) -- winter wheat (Netherlands +
  France, 1990-2021), reproducing [Kallenberg et al. (2023)](https://doi.org/10.1017/eds.2023.28);
  see its own `README.md` for full parameter provenance and a model-fidelity discussion, and
  `examples/nitrogen-winterwheat/PaperRep-Lintul3gym.ipynb` for the matching notebook.

## Finding (or building) more datasets

- [PCSE's own documentation](https://pcse.readthedocs.io/en/stable/) ships further worked-example
  parameter sets alongside its user guide.
- [WUR-AI/PCSE-Gym](https://github.com/WUR-AI/PCSE-Gym) -- the source of this repo's winter-wheat
  dataset -- has additional crop configurations, but most of them (`crops.yaml`, `wheat.yaml`,
  `SUG0601.CAB`) are in **WOFOST** format (`YAMLCropDataProvider`/`CABOFileReader`), for PCSE's
  WOFOST model rather than LINTUL3. They aren't a drop-in for `Lintul3Env` -- which only ever runs
  `pcse.models.LINTUL3` -- without first translating their parameters into the plain
  `PCSEFileReader` `.crop`/`.site`/`.soil` text format used here.
