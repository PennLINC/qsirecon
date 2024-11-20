# Comparisons to other pipelines

Other pipelines for preprocessing DWI data are currently being developed.
Below are tables comparing their current feature sets.
These other pipelines include:

 * [Tractoflow](https://doi.org/10.1016/j.neuroimage.2020.116889)
 * [PreQual](https://doi.org/10.1101/2020.09.14.260240)
 * [MRtrix3_connectome](https://github.com/BIDS-Apps/MRtrix3_connectome)


## Reconstruction

|                           | QSIRecon | Tractoflow | PreQual | MRtrix3_connectome |
| ------------------------- | :-----: | :--------: | :-----: | :----------------: |
| MRTrix3 MSMT CSD          |    ✔    |     ✘      |    ✘    |         ✔          |
| CSD                       | MRtrix3 |    DIPY    |    ✘    |      MRtrix3       |
| Single Shell 3-Tissue CSD |    ✔    |     ✘      |    ✘    |         ✘          |
| DTI Metrics               |    ✔    |     ✔      |    ✔    |         ✔          |
| DSI Studio GQI            |    ✔    |     ✘      |    ✘    |         ✘          |
| MAPMRI                    |    ✔    |     ✘      |    ✘    |         ✘          |
| 3dSHORE                   |    ✔    |     ✘      |    ✘    |         ✘          |

## Tractography

|                                       | QSIRecon | Tractoflow | PreQual | MRtrix3_connectome |
| ------------------------------------- | :-----: | :--------: | :-----: | :----------------: |
| DIPY Particle Filtering               |    ✘    |     ✔      |    ✘    |         ✘          |
| MRtrix3 iFOD2                         |    ✔    |     ✘      |    ✘    |         ✔          |
| Anatomically constrained Tractography |    ✔    |     ✔      |    ✘    |         ✔          |
| DSI Studio QA-Enhanced Tractography   |    ✔    |     ✘      |    ✘    |         ✘          |
| Across-Software tractography          |    ✔    |     ✘      |    ✘    |         ✘          |
| SIFT weighting                        |    ✔    |     ✘      |    ✘    |         ✔          |

## QC

|                               | QSIRecon           | Tractoflow | PreQual | MRtrix3_connectome |
| ----------------------------- | :---------------: | :--------: | :-----: | :----------------: |
| Automated methods boilerplate |         ✔         |     ✘      |    ✘    |         ✘          |
| HTML Reconstruction Report    | NiWorkflows-based |     ✘      | Custom  |         ✘          |
