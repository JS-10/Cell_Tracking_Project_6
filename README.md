# Cell Tracking (Project 6)

## Description

This project, developed as part of the "Computer Vision" course at the University of Cologne, focuses on tracking cancer cells across microscopy image sequences. We evaluate and compare the performance of SORT and DeepSORT for this task, specifically analyzing their ability to handle events like cell divisions and deaths. This project is supervised by Prof. Dr. Kasia Bozek, Dr. Noémie Moreau,  Dr. France Rose and Paul Hahn (M. Sc.).

We use a dataset of videos, each saved as individual JPG frames with COCO-style annotations. Performance is assessed using standard multi-object tracking (MOT) metrics such as MOTA and IDF1.

Our DeepSORT implementation adapts the [DeepSORT with PyTorch](https://github.com/ZQPei/deep_sort_pytorch) repository. We've made modifications for direct hyperparameter and data passing via CLI, and adjusted the tracker to process image frames instead of video streams. Details of these changes are in the cloned repository's [README](/deep_sort_pytorch/README.md) within this project ([here](/deep_sort_pytorch/)).

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
    git clone https://github.com/abewley/sort.git
    ```
    Make sure the `sort` repository is cloned directly into your project's root.

## Usage

This project's primary workflow is managed through the [cell_tracking_hpc.ipynb](/cell_tracking_hpc.ipynb) Jupyter Notebook. This notebook guides you through the process of setting up your data, training the DeepSORT model (if desired), and performing tracking with both DeepSORT and SORT, followed by evaluation.
1.  **Launch Jupyter Notebook and open ```cell_tracking_hpc_ipynb```**:
    ```bash
    jupyter notebook cell_tracking_hpc.ipynb
    ```
2.  **Loading dataset and annotations**
    -   Within the ```cell_tracking_hpc.ipynb``` notebook, locate and set the following variables at the beginning of the relevant sections:
        -   ```DATASET_DIR```: The path to your dataset directory containing the image frames.
        -   ```ANN_FILE```: The path to your COCO-styled annotation JSON file. (Only required to be changed if not using the project's default dataset).
    -   Example:
        ```python
        DATASET_DIR = "path/to/your/unzipped_dataset_jpg_folder"
        ANN_FILE = "path/to/your/annotations.json"
        ```
4.  **Configure and Run DeepSORT Training (Optional)**
    -   Navigate to the DeepSORT training section in the notebook.
    -   Adjust the hyperparameter arrays for ```batch_sizes```, ```lrs``` (learning rates), and ```optimizer``` to test different configurations.
    -   After the training runs, manually select and set the ```best_lr```, ```best_opt```, ```best_batch_size```, and ```best_model``` variables within the notebook to use the optimized parameters for tracking.
5.  **Configure and Run DeepSORT Tracking**
    -   Go to the DeepSORT tracking section in the notebook.
    -   Define arrays for different parameter values you wish to test, such as ```max_age```, ```n_init```, ```min_confidence```, ```max_iou_distance```, ```max_dist```, and ```nms_max_overlap```.
    -   After the tracking process, manually select the best performing values and assign them to ```best_max_age```, ```best_n_init```, ```best_min_confidence```, ```best_max_iou_distance```, ```best_max_dist```, and ```best_nms_max_overlap``` for final evaluation.
6.  **Configure and Run SORT Tracking**
    -   Locate the SORT tracking section in the notebook.
    -   Modify the templates array to define different sets of parameters for SORT, including ```'MAX_AGE'```, ```'MIN_HITS'```, and ```'IOU_THRESHOLD'```. Each dictionary in the templates array represents a unique configuration you wish to test.
        -    Example configuration:
             ```python
             templates = [
                {'name': 'Default',      'MAX_AGE': 1, 'MIN_HITS': 3, 'IOU_THRESHOLD': 0.3},
                {'name': 'Strict_IoU',   'MAX_AGE': 1, 'MIN_HITS': 3, 'IOU_THRESHOLD': 0.5},
                {'name': 'Relaxed_IoU',  'MAX_AGE': 1, 'MIN_HITS': 3, 'IOU_THRESHOLD': 0.1},
                # Add or modify configurations as needed
             ]
             ```

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
