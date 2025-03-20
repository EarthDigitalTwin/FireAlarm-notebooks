# Fire Alarm Demonstration Notebooks

The notebooks in this repo highlight some of the analytics and visualization capabilities of the Fire Alarm: 
Science Data Platform for Wildfire and Air Quality system with the following example use cases:

- [2025 Eaton and Palisades Fires Impacting Southern California](1.%202025%20Eaton%20and%20Palisades%20Fires.ipynb)
- [2023 Canadian Wildfires Impacting New York Air Quality](2.%202023%20Canada%20Wildfires.ipynb)
- [2021 Dixie Wildfire](3.%20Dixie%20Fire.ipynb)
- [2023 Alberta Wildfires](4.%202023%20Alberta%20Wildfires.ipynb)
- [2018 Carr Wildfire](5.%202018%20Carr%20Wildfire.ipynb)
- [Los Angeles ports backlog Fall 2021](6.%20Port%20of%20Los%20Angeles%20Fall%202021.ipynb)
- [Fireworks during 4th of July 2022 in Los Angeles county](7.%202022%20LA%20County%20Fireworks%20July%204th.ipynb)
- [Predicting What We Breathe Los Angeles PM2.5 predictions](8.%20Predicting%20What%20We%20Breathe.ipynb)
- [OCO-3 Snapshot Area Map data](9.%20OCO3.ipynb):
  - Impact of the COVID-19 pandemic response to CO2 emissions
  - Bełchatów Power Station, Poland: Do observed emissions match reported emissions?

## Requirements

* conda >= 22.9.0  

* OS: Mac (more OS options to come)

## Running the notebook

To run the notebooks, run the following commands that create a conda environment called `firealarm_notebook` using the 
`environment.yml` file to include all required dependencies, and install the environment as a kernelspec:
```
conda env create -f environment.yml
conda activate firealarm_notebook
pip install notebook
pip install ipykernel
python -m ipykernel install --user --name=firealarm_notebook
jupyter notebook
```
From the localhost page that opens, you can run the notebooks. Make sure you change the kernel by selecting the option 
at the top Kernel -> Change kernel -> ideas_notebook (see [here](https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments) for more information).

## Adding a new notebook

To add a new notebook, duplicate the `stub_notebook.ipynb` file, name it with the following convention: 
`<notebook number>. <title>.ipynb`. With the notebook implemented in this new file, add a link to it in the list at the 
top of this file.
