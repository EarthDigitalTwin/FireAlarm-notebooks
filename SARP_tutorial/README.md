# FIREALARM SARP TUTORIAL

This repo contains a tutorial notebook for working with SARP data ingested into the FireAlarm platform. Examples include `data discovery`, `querying for data`, and `working with data` to create some simple analysis plots and maps.

## Preparation
In order to be able to run the notebook, you must first prepare your environment by installing some dependencies or Python libraries.

1. Download the repository

- Option A: Clone using git
    ```bash
    git clone https://github.com/EarthDigitalTwin/FireAlarm-notebooks.git
    ```
- Option B: Download ZIP
    - In a web browser, navigate to `https://github.com/EarthDigitalTwin/FireAlarm-notebooks`
    - Click the green Code button, then click Download ZIP
    - Extract the ZIP file to a folder on your computer.

2. Navigate to the SARP folder within the `FireAlarm-notebooks` repo. From the root of the repo:
    ```bash
    cd SARP_tutorial
    ```

3. Create a virtual environment (think walled garden for this code to run in):

- On macOS/Linux:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
- On Windows (Command Prompt):
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

4. Use `pip` to install the list of libraries found in `sarp_requirements.txt` into the virtual environment:
    ```bash
    pip install --upgrade pip
    pip install -r sarp_requirements.txt
    ```
4. Install the venv to be a usable "kernel" in a jupyter notebook (think engine for running the notebook - the libraries we just installed will be available to use in the notebook)
    ```bash
    python -m ipykernel install --user --name=venv --display-name "SARP Tutorial (venv)"
    ```

## Running the notebook

### From an IDE like VS Code
1. Install `Jupyter` extension in VS Code
2. Open `SARP_tutorial.ipynb`
3. Select the kernel we just installed, `SARP Tutorial (venv)` in the top right corner of the notebook. You made need to restart VS Code for it to show up.

### From a browser
1. Start Jupyter notebook server (make sure you've already activated the virtual environment - see above)
```bash
jupyter notebook
```
2. A browser window will likely open displaying a filesystem.
3. Open `SARP_tutorial.ipynb`
4. Select the kernel we just installed, `SARP Tutorial (venv)` by clicking 
```
"Kernel" → "Change kernel" → "SARP Tutorial (venv)"
```