# LINTUL3-Gym

[![PyPI](https://img.shields.io/pypi/v/lintul3-gym)](https://pypi.org/project/lintul3-gym/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)

LINTUL3-Gym is a **transparent, reproducible** [Gymnasium](https://gymnasium.farama.org/)
environment for nitrogen-management reinforcement learning, built on PCSE's LINTUL3 crop model.
Every transition returns a complete, inspectable record -- crop state, weather, and reward
components -- instead of an opaque feature vector, and the bundled winter-wheat example reproduces
a peer-reviewed paper's experiment end-to-end. It ships with a spring-wheat scenario for
`gym.make(...)` to work out of the box, plus two fuller worked examples -- spring wheat and winter
wheat -- with their own tutorial notebooks (see [Examples](#examples)).

## Install

```bash
pip install lintul3-gym
pip install 'lintul3-gym[sb3,viz]'
```

## Quick start

```python
import gymnasium as gym
import lintul3_gym

env = gym.make("Lintul3Gym-v0", decision_interval=7)
observation, info = env.reset(seed=7)
while True:
    observation, reward, terminated, truncated, info = env.step([0.0])
    if terminated or truncated:
        break
```

The action is one nitrogen dose in g N/m². The observation keeps `crop`, `weather`, and `management` fields separate. Each step's `info` includes the names and values consumed by the agent, weather metadata, reward components, and a serializable transition record. `env.unwrapped.history` is the complete episode audit trail.

## Inputs and weather

By default the package loads its bundled spring-wheat `.crop`, `.site`, `.soil`, and `nl1.xlsx` inputs. Point `data_dir` at another directory to use your own inputs: `Lintul3Env` auto-discovers whichever single `*.crop`, `*.site`, and `*.soil` file it finds there *by extension*, not a fixed filename, so any directory shaped like the bundled example works. A sample `.agro` file may also be present for provenance/documentation, but it is never read as an environment input -- see [Custom environments](#custom-environments).

```python
from lintul3_gym import Lintul3Env, WeatherConfig

excel_env = Lintul3Env(data_dir="/path/to/my/inputs")
nasa_env = Lintul3Env(
    weather=WeatherConfig(
        source="nasa",
        locations=((51.97, 5.67), (52.0, 5.5)),
        years=(2004, 2005, 2006),
        random_weather_per_episode=True,
    )
)
```

NASA POWER selections are seeded and cached locally under `~/.cache/lintul3_gym/nasa_power` by default. Set `cache_dir` or `refresh_cache` in `WeatherConfig` to control this behavior.

## Custom environments

`Lintul3Env` is flexible and works with any crop that has a LINTUL3
parameter set, via two pieces of configuration:

- **`data_dir`**: a directory holding exactly one `*.crop`, one `*.site`, and   one `*.soil` file (any filename), and optionally an `nl1.xlsx` for  `WeatherConfig(source="excel")`.
- **`SeasonConfig`**: the crop calendar (`campaign_start`, `crop_start`/`crop_end`  dates and types, `crop_name`, `max_duration`) that would otherwise live in a  PCSE `.agro` file. `Lintul3Env` builds the equivalent calendar from this in  code, so nitrogen application stays entirely under the RL agent's control.

```python
from datetime import date
from lintul3_gym import Lintul3Env, SeasonConfig, WeatherConfig

season = SeasonConfig(
    campaign_start=date(2007, 1, 1), crop_start=date(2007, 1, 1),
    crop_end=date(2007, 9, 1), crop_name="winter-wheat",
    crop_start_type="emergence", crop_end_type="earliest", max_duration=365,
)
env = Lintul3Env(data_dir="examples/envs/winterwheat", season=season, decision_interval=7)
```

Two worked examples ship in this repo:

- [`examples/envs/springwheat/`](examples/envs/springwheat) -- the bundled default (Netherlands,  2006).
- [`examples/envs/winterwheat/`](examples/envs/winterwheat) -- winter wheat (Netherlands + France,  1990-2021), reproducing [Kallenberg et al. (2023)](https://doi.org/10.1017/eds.2023.28); see its  own `README.md` for full parameter provenance and a model-fidelity discussion.

For more LINTUL3-ready datasets, see [PCSE's own documentation](https://pcse.readthedocs.io/en/stable/) and [WUR-AI/PCSE-Gym](https://github.com/WUR-AI/PCSE-Gym) (the source of the winter-wheat set above) -- though most of PCSE-Gym's *other* crop configs are in WOFOST format (`YAMLCropDataProvider`/`CABOFileReader`), not this package's plain-text `PCSEFileReader` `.crop`/`.site`/`.soil` format, so they need translating rather than dropping in directly.

## Reward and comparison

The default reward follows PCSE-Gym's LINTUL cost function:

`storage-organ growth (g/m²) - 10.0 × applied nitrogen (g N/m²)`.

Use `RewardConfig(nitrogen_cost=...)` to change the cost, or `relative_to_zero_nitrogen=True` to score growth relative to a synchronized zero-nitrogen crop run.

## Policies

`lintul3_gym.policies` provides a few simple, inspectable reference policies to evaluate a trained agent against, plus the evaluation helpers used throughout the examples:

- **`ZeroNitrogenPolicy`** -- never applies nitrogen; a lower-bound baseline.
- **`ExpertPolicy`** and **`StandardPracticePolicy`** -- both are *fixed calendar-dose* baselines (apply a predetermined amount of nitrogen on predetermined calendar dates, regardless of crop state): `ExpertPolicy` reproduces the bundled spring-wheat `.agro` schedule (10 and 5 g N/m² on Apr 10 and May 5); `StandardPracticePolicy` reproduces the "Standard Practice" baseline from Kallenberg et al. (2023) for the winter-wheat use case (a configurable total split across three real fertilization dates). Both share the same underlying date-window matching logic.
- **`evaluate_policy(environment, policy, seed=...)`** -- runs one policy episode (works for either of the above, or a Stable-Baselines3 model) and returns a complete, transparent `EvaluationResult`.
- **`evaluate_policy_over_weather(environment, policy)`** -- runs `evaluate_policy` once per `(location, year)` combination in a `WeatherConfig`, round-robin.
- **`evaluate_sb3_policy(vec_env, model, n_episodes=...)`** -- the `VecNormalize`-aware counterpart for trained Stable-Baselines3 models.

See `examples/nitrogen-springwheat/Tutorial-Lintul3gym.ipynb` and `examples/nitrogen-winterwheat/PaperRep-Lintul3gym.ipynb` for these in use.

## Examples

| Example | What it shows |
| --- | --- |
| [`examples/nitrogen-springwheat/Tutorial-Lintul3gym.ipynb`](examples/nitrogen-springwheat/Tutorial-Lintul3gym.ipynb) | Full tutorial -- what LINTUL3/PCSE need as input, how `lintul3_gym` supplies it, and both bundled experiments (train+eval on one fixed season; train+eval across disjoint years/locations via NASA POWER). Ends with a guide to adapting the environment to a different crop. |
| [`examples/nitrogen-winterwheat/PaperRep-Lintul3gym.ipynb`](examples/nitrogen-winterwheat/PaperRep-Lintul3gym.ipynb) + [`examples/envs/winterwheat/README.md`](examples/envs/winterwheat/README.md) | Reproduces Kallenberg et al. (2023)'s winter-wheat nitrogen-management experiment: NASA POWER weather, the paper's train/test locations and years, a Standard Practice baseline, and out-of-distribution climate testing (Netherlands vs. France). |

## How to cite

If you use `lintul3-gym` in your research, please cite it as:

> Ohagwu, C. P. (2026). *LINTUL3-Gym: A transparent, reproducible Gymnasium environment for LINTUL3
> nitrogen management* (Version 0.1.0) [Computer software]. https://github.com/cpohagwu/lintul3-gym

```bibtex
@software{ohagwu_lintul3gym,
  author  = {Ohagwu, Collins Patrick},
  title   = {{LINTUL3-Gym}: A transparent, reproducible {Gymnasium} environment for {LINTUL3} nitrogen management},
  year    = {2026},
  url     = {https://github.com/cpohagwu/lintul3-gym},
  version = {0.1.0}
}
```

If you're using the winter-wheat example to reproduce Kallenberg et al. (2023)'s results, please
also cite that paper -- see [`examples/envs/winterwheat/README.md`](examples/envs/winterwheat/README.md)
for its citation, which is separate from citing this software.

## License

Apache-2.0 (see [`LICENSE`](LICENSE)). The design and economic convention of the default reward
are informed by the GPL-3.0 [PCSE-Gym](https://github.com/WUR-AI/PCSE-Gym) reference project.

The winter-wheat example's data files and patch
([`examples/envs/winterwheat/`](examples/envs/winterwheat)) are copied verbatim from PCSE-Gym and
remain under its GPL-3.0-or-later license rather than this repository's Apache-2.0 license -- see
that directory's `README.md` for details.
