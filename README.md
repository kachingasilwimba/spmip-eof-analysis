---

# SPMIP EOF Analysis

This repository contains code for performing and visualizing **EOF (Empirical Orthogonal Function) analysis** on soil moisture data from the CLM5 (Community Land Model) and ERA5 reanalysis. The goal is to compare how various CLM5 experiments reproduce the dominant EOF modes in reference data (ERA5), focusing on the United States region.

## Overview

1. **Data Sources**  
   - **ERA5**: Reference dataset for land surface variables (resampled to daily).  
   - **CLM5**: Multiple experiments (Exp1, Exp2, Exp3, Exp4[a–d]) with different soil textures or configurations.

2. **Analysis Steps**  
   1. **Anomaly Calculation**: Compute daily anomalies relative to a climatology.  
   2. **EOF Computation**: Use the [eofs Python package](https://github.com/ajdawson/eofs) to extract the leading EOF patterns and principal components.  
   3. **Correlation & Distance Maps**: Compare CLM5 experiments vs. ERA5 via correlation maps and Euclidean distance.  
   4. **Taylor Diagrams**: Summarize model performance in terms of standard deviations and correlation coefficients.  
   5. **Visualization**: Produce consistent subplots for correlation patterns, distance fields, and explained variances.

## Features

- **Custom Taylor Diagram**: A polar plot displaying model standard deviations and correlations vs. the reference.
- **Automated Map Plotting**: Utility functions for contour/pcolormesh with lat/lon grids (using [Cartopy](https://scitools.org.uk/cartopy/docs/latest/)).
- **Masking & Weighting**: Examples of applying US-region masks, sqrt(cos(lat)) weighting, etc.
- **Organized Scripts**:
  - **`eofcalc_pcnorm`**: Perform EOF with normalized PCs.
  - **`calculate_anomaly`**: Daily anomalies from day-of-year or other groupings.
  - **`plot_map_with_threshold`**: Overlays hatches for areas under a certain threshold (e.g., significance).
  - **`create_taylor_diagram`**: High-level function constructing Taylor Diagrams.

## Requirements

1. **Python 3.8+** (or compatible version)  
2. **Dependencies** (install via `conda` or `pip`):
   - [xarray](https://docs.xarray.dev/en/stable/)
   - [numpy](https://numpy.org/)
   - [matplotlib](https://matplotlib.org/)
   - [cartopy](https://scitools.org.uk/cartopy/docs/latest/)
   - [eofs](https://github.com/ajdawson/eofs)
   - [scipy](https://www.scipy.org/)  
3. **Optional**:
   - [Jupyter Notebook](https://jupyter.org/) for interactive exploration.

## Installation

Clone this repository:
```bash
git clone https://github.com/your-username/spmip-eof-analysis.git
cd spmip-eof-analysis
```

Create or activate an environment with the required libraries:
```bash
conda create -n spmip-eof python=3.9
conda activate spmip-eof
conda install xarray numpy matplotlib cartopy eofs scipy
```

## Usage

1. **Data Preprocessing**:  
   Make sure you have the **ERA5** and **CLM5** NetCDF data files in a directory accessible to the scripts. Adjust paths in the code (e.g., `path = "/glade/work/.../"`).

2. **Running EOF Scripts**:  
   - **`python script_eof_analysis.py`** (example) to compute daily anomalies and EOF modes.
   - **`python script_plot_correlation.py`** to generate correlation maps for multiple experiments.
   - **`python script_taylor_diagram.py`** to produce Taylor Diagrams summarizing model performance.

3. **Outputs**:  
   - **Figures**: PDF or PNG files in the `./Figures/` folder.
   - **Intermediate NetCDFs** (optional): Sliced or aggregated data if you choose to save them.

## Project Structure

```
spmip-eof-analysis/
├── data/
│   ├── ERA5_Land.nc
│   ├── CLM5_SPMIP_Exp1_1980-2010.nc
│   └── ...
├── Figures/
│   └── ... (generated plots)
├── scripts/
│   ├── eof_utilities.py
│   ├── plot_util.py
│   ├── Taylordiagram.py
│   ├── taylor_diagram.py
│   └── ...
├── README.md
└── environment.yml (optional conda environment file)
```

## Contributing

Contributions are welcome! If you find a bug or have a feature request:
1. Fork this repo.
2. Create a new branch: `git checkout -b feature/my-new-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push the branch: `git push origin feature/my-new-feature`
5. Submit a Pull Request.

## License

[MIT License](LICENSE).  
See the `LICENSE` file for details.
## References

- **EOF Analysis in Python**: For a comprehensive Python package for EOF analysis, consider the `eofs` package by Andrew Dawson. It offers a robust implementation suitable for large datasets and includes metadata handling.

  Repository: [https://github.com/ajdawson/eofs](https://github.com/ajdawson/eofs)

- **EOFs Cookbook**: The Project Pythia EOFs Cookbook provides tutorials and examples on EOF analysis applications in climate data.

  Repository: [https://github.com/ProjectPythia/eofs-cookbook](https://github.com/ProjectPythia/eofs-cookbook)
  
-  [Taylor Diagrams](https://gist.github.com/ycopin/3342888)
---
## Acknowledgments

This project was inspired by existing EOF analysis tools and methodologies. Special thanks to the contributors of the `eofs` package and the EOFs Cookbook for their valuable resources. 

### Contact

For questions or further discussion, reach out to Kchinga Silwimba at kachingasilwimba@u.boisestate.edu. If you publish results based on this code, please provide appropriate citation or reference to this repository.
