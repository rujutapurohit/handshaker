# handshaker

Photometric corrections (`magcor`) for Roman WFI calibration effects.

```bash
pip install -e .
handshaker configs/example.yaml 
handshaker configs/example.yaml --mode polyfit -o out/example_polyfit
pytest
```

## Two modes

- **`magcor`** — per observation and per effect (`shift`, `ice`, `total`), the
  magnitude correction `mag(perturbed bandpass) − mag(nominal)` on the SN SED. If
  a calibration-star SED is configured, the reported magcor is the SN-minus-star
  **chromatic differential** (per DES); otherwise it's the SN-only shift. Writes
  `magcor.csv`.
- **`polyfit`** — fits the throughput ratio `T(λ)/T_ref(λ)` with an Nth-order
  polynomial per (band, SCA) and writes the `ROW SCA ICE_THICK BAND WAVE
  RATIO_TRUE RATIO_POLY RESIDUAL` table plus a `polyfit_summary.csv` with the
  coefficients, wavecor, and residual RMS.

The config file is copied into the output dir (`config.used.yaml`) on every run.

## Ice model

Ice changed throughput by a thickness-dependent multiplicative factor
`T(t)/T(t=0)`. The **released** model (`ice: {flag: thushara}`) reads the
`spectral_responses_SCA*.ecsv` response (columns `T_ratio_d<thickness>nm`) and
linearly interpolates it, per wavelength, to the requested thickness. Ice thickness
per observation comes from the `ice_thickness_nm` column. The toy
deposition-vs-(position, MJD) model (`ice_rep_sim.py`, needs the growth-rate
mosaic `.npz`) can supply a thickness instead via
`ReleasedIceModel.thickness_from_position`, but is not required for the default
per-observation-thickness workflow. `ice: {flag: none}` disables ice.

Thushara's model: https://github.com/RomanSpaceTelescope/ice-toy-model 

## Layout

```
src/handshaker/
  cli.py         `handshaker <config.yaml> [--mode] [--output-dir]`
  pipeline.py    config -> objects -> magcor/polyfit tables
  config.py      YAML config load + copy-to-output
  observation.py per-observation input record
  filters.py     FilterSource: synthetic / roman_dir (rujuta,thushara) / kcor / soc(stub)
                 + FULL/PERTURBATION Transmission and compose()
  sed.py         SEDModel: synthetic blackbody + SALT3SEDModel (+ x1,c fit)
  synphot.py     synthetic magnitude + effective wavelength
  ice.py         released ice model from the response ecsv (+ none)
  magcor.py      mode 1: per-effect + total magcor (SN, and SN-minus-star)
  polyfit.py     mode 2: T/T_ref polynomial fit + residual table
  fpa.py         SCA / focal-plane check (NullFocalPlane today)
  data/spectral_responses_SCA10.ecsv   bundled released ice response
configs/
  example.yaml       runnable with NO external data
  roman_salt3.yaml   production: SALT3 + ROMAN_FILTERS on Midway
tests/test_smoke.py  no-data regression tests
```

## Running on Midway (real data)

`configs/roman_salt3.yaml` is the production config; edit its paths for your run.
It needs `$SNDATA_ROOT` set (with `models/SALT3/SALT3.P22-NIR`) and the
`ROMAN_FILTERS` directory laid out as `ROMAN_<band>/ROMAN_<band>[_shifted_<sca>].dat`.

```bash
# on Midway, in your env: 
cd handshaker
pip install -e .
export SNDATA_ROOT=/project2/rkessler/PRODUCTS/SNDATA_ROOT   # if not already set
handshaker configs/roman_salt3.yaml -v
```

## Status

Runs end-to-end today: both modes, synthetic + SALT3 SEDs, synthetic +
ROMAN_FILTERS + kcor filter sources, released ice model, config reproducibility.

Still stubbed / TODO: the SOC filter format, the focal-plane `(RA,DEC)->SCA`
projection (so `fpa` only trusts the supplied SCA), the ice deposition mosaic
`.npz` wiring, and the polyfit *magnitude-level* validation
(`mag_err = mag_poly − mag_true`) called out in the spec.

- Need to find better ways to write modelspec.root files (alternative to root)
- Better logging and debugging utils

See `Handshaker_Magcor.pdf` for the full spec.
