## Installation

- Install postgresql and set posgresql password as environment variable
```bash
# For windows (restart terminal after running the command)
setx PGPASSWORD "your_postgres_password"

# For linux and MacOS
export PGPASSWORD="your_postgres_password"
```
- Clone the repository
- Download the  `dataset.json` file into the root of the project
- Create python environment
```bash
python -m venv .venv
```
- Enter the environment
```bash
# For windows (command prompt)
.\.venv\Scripts\activate

# For windows (powershell)
.\.venv\Scripts\Activate.ps1

# For Linux and MacOS
./.venv/bin/activate
```
- Install uv and other packages
```bash
pip install uv
uv sync
```
- Setup database
```bash
python main.py setup
```
- Ingest the dataset
```bash
# Make sure dataset.json is in the current directory
python main.py ingest
```
