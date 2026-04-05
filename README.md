# Train Delay Analysis: Network Connectivity and Delay Propagation

## Project Overview

This project analyzes train travel data to understand station connectivity patterns and delay propagation across a railway network. Using adjacency matrices and delay analysis, the notebook visualizes how stations are connected and quantifies average delays between stations.

## Features

- Construction of connectivity matrices from travel records
- Visualization of station networks using heatmaps
- Analysis of average delays between connected stations
- Threshold-based filtering for significant delays
- Identification of delay propagation patterns

## Prerequisites

Before running this notebook, ensure you have the following installed:

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Required Python packages (see Installation section)

## Project Structure

### Python Modules
- `SampleWork.py` - Main analysis module (CVaR, distributions)
- `correlation.py` - Correlation analysis between variables
- `presentation_figure.py` - Generate publication-ready figures

### Jupyter Notebooks
- `main_v2.ipynb` - Main analysis pipeline
- `draw_cvar.ipynb` - CVaR visualization notebook
- `UniquePath.ipynb` - Unique path identification
- `path_finder_for_one_journey.ipynb` - Route optimization
- `get_sorted_journey_from_data_source.ipynb` - Data preprocessing

### Data Files
- `acm_to_blg_corrected_delays.csv` - Corrected delay data
- `adjecencyMatrix.xlsx` - Station connectivity matrix
- `bootstrap_results.tex` - Bootstrap statistical results
```

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Railway_Network_RSP.git
cd Railway_Network_RSP


## Launch Jupyter Notebook

## Run Analysis
Open main_v2.ipynb for complete analysis pipeline
Open draw_cvar.ipynb for risk visualization
Open path_finder_for_one_journey.ipynb for route optimization

## Key Results
Identified delay patterns across different routes
Calculated CVaR at 95% confidence for risk assessment
Validated findings using bootstrap methods

## Author
Achini Nanayakkara
Department of Data Science
Uppsala University, Sweden
