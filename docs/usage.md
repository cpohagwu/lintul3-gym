# Usage

## Environment construction

`Lintul3Env(data_dir=None, *, season=None, weather=None, reward=None, decision_interval=7,
max_nitrogen=20.0, render_mode=None)` builds one LINTUL3 crop season per episode. With no arguments
it loads the packaged 2006 spring-wheat scenario (Netherlands, bundled Excel weather).

- **`data_dir`** -- a directory containing exactly one `*.crop`, one `*.site`, and one `*.soil`
  file (any filename -- discovered by extension, not a fixed name), and optionally an `nl1.xlsx`
  for Excel weather. See [Custom environments](custom-environments.md) for the full pattern,
  including the two worked examples shipped under `examples/envs/`.
- **`season`** -- a `SeasonConfig` describing the crop calendar (campaign start, crop start/end
  dates and types, `max_duration`). This replaces a PCSE `.agro` file's `CropCalendar` section;
  nitrogen application stays entirely under the agent's control -- an `.agro` file's fertilizer
  schedule, if present, is never read as an environment input.
- **`weather`** -- a `WeatherConfig`. `source="excel"` (the default) reads the data directory's
  bundled `nl1.xlsx` for one fixed season. `source="nasa"` uses PCSE's NASA POWER provider across
  any `locations`/`years` you supply: set `random_weather_per_episode=True` to sample a seeded
  `(location, year)` pair every reset, or leave it `False` to deterministically round-robin
  through every combination in order -- useful for held-out evaluation, since a fixed order lets
  you line up the same combination across different policies. NASA POWER responses are cached to
  disk under `~/.cache/lintul3_gym/nasa_power` by default (override with `cache_dir`/
  `refresh_cache`).
- **`reward`** -- a `RewardConfig`. The default reward is storage-organ growth minus
  `nitrogen_cost * applied nitrogen` (10.0 g/m² by default); `relative_to_zero_nitrogen=True` (the
  default) scores growth relative to a synchronized zero-nitrogen crop run of the same season,
  matching PCSE-Gym's reward convention.
- **`decision_interval`** -- days simulated per `step()` call (default 7, i.e. weekly decisions).
- **`max_nitrogen`** -- the upper bound of the continuous `Box` action space, in g N/m².

## Observation and action spaces

The observation is a `gym.spaces.Dict` with three separately-labeled groups -- `crop` (the 9
LINTUL3 state variables in `CROP_FEATURES`), `weather` (4 driving variables in `WEATHER_FEATURES`),
and `management` (cumulative nitrogen and the last action) -- rather than one opaque flattened
vector. The action space is a one-element `Box(0.0, max_nitrogen)`: a single nitrogen dose in
g N/m² per `step()`.

## Stable-Baselines3 adapters (`lintul3_gym.sb3`)

- **`make_sb3_env(environment)`** -- flattens the `Dict` observation into crop/weather/management
  order for `MlpPolicy`-style algorithms, keeping the continuous `Box` action space.
- **`make_discrete_sb3_env(environment)`** -- as above, plus `DiscretizeNitrogen`, exposing a
  fixed `{0, 20, 40}` kg N/ha `Discrete(3)` menu (matching the discrete action space used in the
  literature this package's winter-wheat example reproduces).
- **`make_training_env(environment, log_dir=...)`** -- wraps the discretized environment in
  `Monitor` + `DummyVecEnv` + `VecNormalize` for training.
- **`make_eval_env(environment, vecnormalize_path, log_dir=...)`** -- rebuilds the same wrapper
  stack with the *saved* normalization statistics frozen, so evaluation sees the same observation
  scale training did. Always evaluate a trained model through this, not a raw environment or a
  freshly-constructed `VecNormalize` -- see the "why not just `model.predict()`" explanation in
  `examples/nitrogen-springwheat/Tutorial-Lintul3gym.ipynb` for the two ways that shortcut fails
  silently.

## Visualization (`lintul3_gym.viz`, optional `[viz]` extra)

`plot_comparison`, `plot_complete_comparison_yearly`, and `plot_complete_comparison_monthly` plot
one or more policies' `EpisodeRecord` histories against each other (single-season, multi-season
side-by-side, and month-aggregated median-with-interval views, respectively).
`plot_nitrogen_vs_rainfall` scatters total nitrogen applied against average seasonal rainfall for
a fixed-dose baseline and a median+IQR ensemble policy, one point per test year -- see
`examples/nitrogen-winterwheat/PaperRep-Lintul3gym.ipynb` for it in use.

## Known issue: noisy PCSE logging on Windows

Importing `pcse` (a side effect of building any `Lintul3Env`) configures a shared root-logger
`RotatingFileHandler` writing to `~/.pcse/logs/pcse.log`. On Windows, if more than one process has
ever imported `pcse` (e.g. two Jupyter kernels), that handler's log-rotation rename can fail with a
`PermissionError`, which Python's own `logging` module catches and reports as a harmless but noisy
`--- Logging error ---` banner -- the simulation itself is unaffected either way; only the one log
line that triggered the failed rotation is lost. `lintul3_gym` never touches PCSE's logging
configuration automatically, but ships an explicit, opt-in fix:

```python
from lintul3_gym import silence_pcse_log_rotation

silence_pcse_log_rotation()
```

Call this once (idempotent on repeat calls) to replace the rotating handler with a plain,
non-rotating one, so the failing rename is never attempted again. Both tutorial notebooks call
this in their "Setup Environment" section; remove that cell if you'd rather see PCSE's logging
behave exactly as it documents.
