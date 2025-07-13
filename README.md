# Cell Tracking (Project 6)

This project was developed as part of the course "Computer Vision" at the University of Cologne and supervised by Prof. Dr. Kasia Bozek, Dr. Noémie Moreau,  Dr. France Rose and Paul Hahn (M. Sc.).

## Description

[...]

## Requirements

Before proceeding with the installation, ensure your system meets the following requirements:

* **Python:** Version 3.9 or higher.
* **Disk Space:** if using the same dataset, At least 20GB of free disk space for the unzipped dataset_jpg.
* **Performance Requirements:** We used [RAMSES](https://itcc.uni-koeln.de/hpc/hpc/ramses) to do high performance computations like training our DeepSORT model.

**Note:** If you plan to utilize GPU acceleration, an NVIDIA GPU with CUDA Toolkit installed is required. 
    
## Installation

To get started, please follow these steps to install the necessary requirements:

1.  **Install Python Dependencies:**
    Navigate to the root directory of this project and install the required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Clone the SORT Repository:**
    This project relies on the Simple Online and Realtime Tracking (SORT) repository. Clone it into your project directory using the following command:
    ```bash
    git clone [https://github.com/abewley/sort.git](https://github.com/abewley/sort.git)
    ```
    Make sure the `sort` repository is cloned directly into your project's root.

## Usage

[...]

## Feedback and Contributions

We welcome feedback, bug reports, and contributions! If you encounter any issues or have suggestions for improvement, please [open an issue](https://github.com/JS-10/Cell_Tracking_Project_6/issues/new) or [submit a pull request](https://github.com/JS-10/Cell_Tracking_Project_6/pulls).

### Contributors

These are the current contributors for this repository. Feel free to contribute!

<a href="https://github.com/JS-10/Cell_Tracking_Project_6/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=JS-10/Cell_Tracking_Project_6" />
</a>

## Contact

For any further inquiries or assistance, feel free to contact us!

**GitHub Profiles:** 
- [Maximilian Karhausen](https://github.com/m4p4k4)
- [Johannes Simon](https://github.com/JS-10)

## Third-Party Code and Licenses
This project is available under the [GNU General Public License v3.0 (GPLv3)](LICENSE) and uses the following third-party code:

### [SORT](https://github.com/abewley/sort) — GPLv3 License  
This project will include source code from `SORT` for multi-object tracking (MOT). 
It is licensed under the GNU General Public License v3.0 (GPLv3).
See the [SORT license](https://github.com/abewley/sort/blob/master/LICENSE) for more information.

### [DeepSORT with PyTorch](https://github.com/ZQPei/deep_sort_pytorch) - MIT License
This project includes components from `deep_sort_pytorch` for appearance-based MOT.
It is licensed under the MIT License.
See the [deep_sort_pytorch license](https://github.com/JS-10/Cell_Tracking_Project_6/blob/main/deep_sort_pytorch/LICENSE) for more information.

We have made several adaptations to the original code to fit our dataset and project structure. For details on our changes, see the [adapted README](https://github.com/JS-10/Cell_Tracking_Project_6/blob/main/deep_sort_pytorch/README.md).
