# Contributing to obnb

First and foremost, we thank you for your interest in contributing to the `obnb` pacakge! We welcome any kind of contribution, e.g.,

- New features
- Bug fixes and new tests
- Typo fixes or improving the documentation in general

To get started,

1. Make a fork of the repository, and create a new branch where you will be making your changes
1. After you are done with the new changes, make a Pull Request https://github.com/krishnanlab/obnb/pulls of your changes

## Dev notes

### Installation

To develop `obnb`, first clone your forked repository to your machine (replace `<your_github_handle>` with your GitHub handle)

```bash
git clone https://github.com/<your_github_handle>/obnb
cd obnb
```

If you have done it before, make sure your fork is in-sync with the latest changes and pull the changes to your local copy

```bash
git pull
```

Then, create a conda environment for developing `obnb` (this is not required, but it is recommended to have an isolated development
environment as it keeps the dependency versions clean)

```bash
conda create -n obnb-dev python=3.9
conda activate obnb-dev
```

Install `obnb` with development dependencies

```bash
pip install -r requirements.txt  # install dependencies with pinned version
pip install -e ".[dev]"  # install extra dependencies for dev
```

Finally, install pre-commit hooks to enable automated code quality control for each commit

```bash
pre-commit install
```

### Testing

To run all existing unit-tests, simply run `pytest` at the project root directory

```bash
pytest
```

### Linting

We provide some additional code quality control tools via `tox` to help catch early mistasks:

```bash
tox -e mypy,flake8
```

### Continuous integration

We have set up continuous integration via [GitHub Actions](https://github.com/krishnanlab/obnb/actions).
Every time a pull request is made from a branch (and any following push to that branch) will trigger automated [testing](#testing)
and [linting](#linting) to check the changes against out dev guidelines.

## Building documentation

1. Install `obnb` with doc dependencies (e.g., [Sphinx](https://www.sphinx-doc.org/en/master/))

   ```bash
   pip install -e .[full,doc]
   ```

1. Install [PyTorch](https://pytorch.org/get-started/locally/) if you have not done so yet, or simply install a
   minimal cpu version of PyTorch if you only want to build the `obnb` documentation:

   ```bash
   conda install pytorch cpuonly -c pytorch
   ```

1. Build the documentation

   ```bash
   cd docs
   make html
   ```

You can now view the documentation page by opening `docs/build/html/index.html`

## Data preparation and releasing notes

First, bump data version in `__init__.py` to the next data release version, e.g., `obnbdata-v0.1.0 -> obnbdata-v0.1.1-dev`.
Then, download and process all latest data by running

```bash
python script/release_data.py
```

By default, the data ready to be uploaded (e.g., to [Zenodo](zenodo.org)) is saved under `data_release/archived`.
After some necessary inspection and checking, if everything looks good, upload and publish the new archived data.

**Note:** `dev` data should be uploaded to the [sandbox](https://sandbox.zenodo.org/record/1097545#.YxYrqezMJzV) instead.

Check items:

- [ ] Update `__data_version__`
- [ ] Run [`release_data.py`](script/release_data.py)
- [ ] Upload archived data to Zenodo (be sure to edit the data version there also)
- [ ] Update url dict in config (will improve in the future to get info from Zenodo directly)
- [ ] Update network stats in data [test](test/test_data.py)

Finally, commit and push the bumped version.
