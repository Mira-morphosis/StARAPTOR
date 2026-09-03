# StARAPTOR
*Steam Automated Review Association Miner for Player-count Time-series Optimized Regression*


A configurable, scalable, and self-hostable tool for game developers to analyze Steam reviews, discover associations between player characteristics and reported issues, and identify review patterns associated with player-count changes. The selected KPI-relevant association rules, together with review-derived features, are then used to train an optimized regressor that forecasts daily mean concurrent player counts over the following 30 days.

>*Note: This README is a WIP, as the project is at an early stage.*

## Getting Started

### Prerequisites & System Dependencies

Ensure you have Python **3.13** installed along with `pipx` (recommended for installing CLI tools like Poetry) and essential build tools for C/C++ extensions required by data science packages (such as XGBoost, DuckDB, or Optuna).

#### 1. Install System Dependencies & Python 3.13

Select the instructions for your Linux distribution:

* **Debian / Ubuntu** (Ubuntu 24.04+ or Debian 13+ support Python 3.13 natively):
  `sudo apt update`
  `sudo apt install -y python3.13 python3.13-venv python3-pip pipx build-essential git`

* **Fedora / RHEL / openSUSE**:
  * **Fedora**:
    `sudo dnf install -y python3.13 python3.13-devel pipx gcc gcc-c++ make git`
  * **openSUSE Tumbleweed**:
    `sudo zypper install -y python313 python313-devel python313-pipx gcc gcc-c++ git`

* **Arch Linux / Manjaro**:
  `sudo pacman -Syu --needed python python-pipx base-devel git`

---

### Installation & Workspace Setup

#### 1. Install Poetry
The recommended way to install Poetry isolated from system packages is via `pipx`:

`pipx install poetry`

Note: Make sure `pipx` binaries are added to your PATH by running `pipx ensurepath` and restarting your terminal session.
>Make sure that installed Poetry version is 2.x or greater, as Poetry 1.x does not support submodules.

#### 2. Clone the Repository
`git clone https://github.com/your-username/staraptor.git`
`cd staraptor`

#### 3. Install Project Dependencies & Local Modules
Poetry 2.x natively handles local directory dependencies without requiring external workspace plugins. Simply run `poetry install` in the root directory to create the virtual environment and link all sub-packages (`libs/storage`, `libs/analyzer`, etc.) in editable mode:

`poetry install`

#### 4. Configure Environment Variables
Copy or create a `.env` file in the project root to set up required API keys or environment flags:

`cp .env.example .env`  # Edit .env with your actual parameters/API keys

#### 5. CLI Executable Link (Optional)
The repository includes a pre-configured `staraptor` executable script wrapper in the root directory. You can optionally create a symlink to run `staraptor` from anywhere on your system:

`ln -s "$(pwd)/staraptor" ~/.local/bin/staraptor`

---

### Running StARAPTOR

You can execute StARAPTOR using the `./staraptor` wrapper script (or just `staraptor` if added to your PATH).

#### 1. Initial Database Cold-Start (`--init`)
Runs the full initialization pipeline for a Steam App ID: fetches historical KPIs & reviews, mines association rules, and performs initial XGBoost hyperparameter tuning:
default
`./staraptor --app-id 1086940 --init`

Options:
* Add `--skip-tuning` to bypass XGBoost hyperparameter optimization during initialization:
  `./staraptor --app-id 1086940 --init --skip-tuning`
* Add `--max-days <DAYS>` to limit historical lookback period (e.g., last 30 days):
  `./staraptor --app-id 1086940 --init --max-days 30`

#### 2. Run Daily Update Once (`--run-once`)
Runs a single update cycle (fetches recent KPIs, today's reviews, re-mines rules, and checks parameter staleness):

`./staraptor --app-id 1086940 --run-once`

#### 3. Continuous Scheduler Daemon (`--daily`)
Launches a continuous background daemon that runs an update cycle every 24 hours (or a custom interval specified by `--interval`):

`./staraptor --app-id 1086940 --daily --interval 24`

#### 4. Hyperparameter Tuning (`--tune`)
Runs XGBoost hyperparameter optimization via Optuna:

##### Tune only if parameters are stale or new data tiers are reached
`./staraptor --app-id 1086940 --tune`

##### Force re-tuning across all data tiers regardless of staleness
`./staraptor --app-id 1086940 --tune --force --n-trials 200`

#### 5. Force Mining (`--force-mining`)
Clears existing mined rules in DuckDB and re-computes historical association rules using sliding window analysis:

`./staraptor --app-id 1086940 --force-mining`

#### 6. Export Rules (`--export`)
Exports all mined rules for the specified App ID from DuckDB to CSV format:

`./staraptor --app-id 1086940 --export`
---
### Configuring StARAPTOR
Configuration is done by editing `.env`. 
Consult the file for an explanation of what environment variables the system requires (and what values are set by default).