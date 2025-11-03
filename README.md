# The ```sailsprep``` repo

[![Build](https://github.com/sensein/sailsprep/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/sensein/sailsprep/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/sensein/sailsprep/branch/main/graph/badge.svg?token=2V7LMSZ1DZ)](https://codecov.io/gh/sensein/sailsprep)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![pages](https://img.shields.io/badge/api-docs-blue)](https://sensein.github.io/sailsprep)

Welcome to the ```sailsprep``` repo! This is a Python repo for doing incredible video-based human pose estimation analyses. **STAY TUNED!**

**Caution:**: this package is still under development and may change rapidly over the next few weeks.
## General information

To manage dependencies, this project uses Poetry. Make sure you've got poetry installed.
On Engaging, you need to first run at the root of the repo :
```
module load miniforge
pip install poetry
poetry install
```

## Preprocessing
### BIDS-conversion
The conversion pipeline requires FFmpeg ≥ 6.0 compiled with the vidstab library.
Because FFmpeg compiled with vidstab is not a Python package, it must be installed separately.
You'll need to run (outside any environment):

```
cd ~
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar -xJf ffmpeg-release-amd64-static.tar.xz
mv ffmpeg-*-static ffmpeg_static
export PATH="$HOME/ffmpeg_static:$PATH"

```

To make this permanent, add the last line to your ~/.bashrc or ~/.bash_profile.
You can verify that FFmpeg has the right version (≥ 6.0):
```
ffmpeg -version
```
You'll need to submit the script on Engaging using sbatch. We've
provided the sumbission files so you'll simply need to run (with module miniforge deactivated) :
```
jid=$(sbatch --parsable jobs/run_bids_convertor.sh)
sbatch --dependency=afterok:$jid jobs/merge_cleanup.sh
```
This will convert the raw video into BIDS format in a clean fashion.
## Features
- A few
- Cool
- Things
- These may include a wonderful CLI interface.

## Installation
Get the newest development version via:

```sh
pip install git+https://github.com/sensein/sailsprep.git
```

## Quick start
```Python
from sailsprep.app import hello_world

hello_world()
```

## Contributing
We welcome contributions from the community! Before getting started, please review our [**CONTRIBUTING.md**](https://github.com/sensein/sailsprep/blob/main/CONTRIBUTING.md).


### To do:
- [ ] A
- [ ] lot
- [ ] !
