The `AirQuality_demo.ipynb` notebook highlights some of the analytics and visualization capabilities of Fire Alarm: Science Data Platform for Wildfire and Air Quality with six use cases:
- 2023 Canadian Wildfires Impacting New York Air Quality
- 2021 Alisal Wildfire
- 2021 California Wildfires
- 2018 Carr Wildfire
- Los Angeles ports backlog Fall 2021
- Fireworks during 4th of July 2022 in Los Angeles county
- Predicting What We Breathe Los Angeles PM2.5 predictions

__Requirements__  

* conda >= 22.9.0  

* OS: Mac (more OS options to come)

__Running the notebook__  

To run the `AirQuality_demo.ipynb` notebook, run the following commands that create a conda environment called `firealarm_notebook` using the `environment.yml` file to include all required dependencies, and install the environment as a kernelspec:
```
conda env create -f environment.yml
conda activate firealarm_notebook
pip install notebook
pip install ipykernel
python -m ipykernel install --user --name=firealarm_notebook
jupyter notebook
```
From the localhost page that opens, you can run the ideas notebook. Make sure you change the kernel by selecting the option at the top Kernel -> Change kernel -> ideas_notebook (see [here](https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments) for more information).
