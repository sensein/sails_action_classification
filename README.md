# The ```sailsprep``` repo

[![Build](https://github.com/sensein/sailsprep/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/sensein/sailsprep/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/sensein/sailsprep/branch/main/graph/badge.svg?token=2V7LMSZ1DZ)](https://codecov.io/gh/sensein/sailsprep)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![pages](https://img.shields.io/badge/api-docs-blue)](https://sensein.github.io/sailsprep)

Welcome to the ```sailsprep``` repo! This is a Python repo for doing incredible video-based human pose estimation analyses. **STAY TUNED!**

**Caution:**: this package is still under development and may change rapidly over the next few weeks.

This will convert the raw video into BIDS format in a clean fashion.
## Features
- A few
- Cool
- Things
- These may include a wonderful CLI interface.

## Installation
To manage dependencies, this project uses Poetry. Make sure you've got poetry installed.
On Engaging, you need to first run at the root of the repo :
```
module load miniforge
pip install poetry
poetry install
```

The BIDS-conversion tool of sailsprep requires FFmpeg ≥ 6.0 compiled with the vidstab library.
Because FFmpeg compiled with vidstab is not a Python package, it must be installed separately.
You'll need to run (outside any environment):

```
cd ~
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar -xJf ffmpeg-release-amd64-static.tar.xz
mv ffmpeg-*-static ffmpeg_static
export PATH="$HOME/ffmpeg_static:$PATH"

```

Get the newest development version via:

```sh
pip install git+https://github.com/sensein/sailsprep.git
```
## Quick start

Tools developped in sailsprep
|Tool|Documentation|
|----|--------------|
|BIDS-conversion| [link to documentation](docs/BIDS_convertor.md)


## Contributing
We welcome contributions from the community! Before getting started, please review our [**CONTRIBUTING.md**](https://github.com/sensein/sailsprep/blob/main/CONTRIBUTING.md).


### To do:
- [ ] A
- [ ] lot
- [ ] !
