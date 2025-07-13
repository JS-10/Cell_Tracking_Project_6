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

1.  **Clone this Repository:**
    ```bash
    git clone https://github.com/JS-10/Cell_Tracking_Project_6.git
    ```

2.  **Install Numpy and Scipy:**
    Navigate to the root directory of this project and install numpy and scipy first by using pip:
    ```bash
    pip install numpy~=1.23.5
    pip install scipy~=1.10.1
    ```

3.  **Install Python Dependencies:**
    In the same root directory, install the required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install PyTorch Dependencies:**
    In the same root directory, install the PyTorch dependencies independently to have access to GPUs, using the following pip command:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ```

5.  **Clone the SORT Repository:**
    This project relies on the Simple Online and Realtime Tracking (SORT) repository. Clone it into your project directory using the following command:
    ```bash
    git clone https://github.com/abewley/sort.git
    ```
    Make sure the `sort` repository is cloned directly into your project's root.

## Usage

This project's primary workflow is managed through the [cell_tracking_hpc.ipynb](/cell_tracking_hpc.ipynb) Jupyter Notebook. This notebook guides you through the process of setting up your data, training the appearance model of DeepSORT, and performing tracking with both DeepSORT and SORT, followed by evaluation.
1.  **Launch Jupyter Notebook and open ```cell_tracking_hpc.ipynb```**:
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
4.  **Train the Appearance Model of DeepSORT**
    -   **After** running the code to crop the cells, navigate to the DeepSORT training section in the notebook.
    -   Training is performed using CLI commands that invoke the `train.py` script inside the cloned DeepSORT repo (`./deep_sort_pytorch/deep_sort/deep/train.py`). These commands are run from the notebook using `!python`.
    -   If you want to **test and compare different configurations**, adjust the hyperparameter arrays for ```batch_sizes```, ```lrs``` (learning rates), and ```optimizer```.
    -   After running each test, manually select and set the ```best_lr```, ```best_opt``` and ```best_batch_size``` variables within the notebook to use the optimized parameters for final tracking.
    -   If you want to **skip testing and directly train an appearance model**, just run the cells with the pre-selected values (or change the values as you like, does not guarantee good results though) for ```best_lr```, ```best_opt``` and ```best_batch_size```, and for training the appearance model, run the final cell before it shows the loss plot.
    -   Example command that you **do not need to run manually, as the notebook builds and runs it for you**:
        ```python
        !python deep_sort_pytorch/deep_sort/deep/train.py --data-dir cell_crops --epochs 60 --batch_size {best_batch_size} --lr {best_lr} --optimizer {best_opt} --model_name {model_name}
        ```
    -   After training, you can find all appearance models as `.pth` files in the directory `deep_sort_pytorch/deep_sort/deep/checkpoint`.
5.  **Configure and Run DeepSORT Tracking** 
    -   **After** training the appearance model, navigate to the DeepSORT tracking section in the notebook.
    -   Tracking with DeepSORT is also performed using CLI commands; here they invoke the `deepsort.py` script inside the cloned DeepSORT repo (`./deep_sort_pytorch/deepsort.py`). These commands are run from the notebook using `!python`.
    -   For the parameter ```best_model```, manually select the best-performing appearance model from training before.
    -   Example usage (same as in the respective cell):
        ```python
        best_model = 'deep_sort_pytorch/deep_sort/deep/checkpoint/model_59.pth'
        ```
    -   If you want to **test and compare different configurations**, define arrays for different hyperparameter values in the same cell where you select the appearance model, such as ```max_age```, ```n_init```, ```min_confidence```, ```max_iou_distance```, ```max_dist```, and ```nms_max_overlap```.
    -   After tracking with each configuration, manually select the best performing values and assign them to ```best_max_age```, ```best_n_init```, ```best_min_confidence```, ```best_max_iou_distance```, ```best_max_dist```, and ```best_nms_max_overlap``` for final evaluation.
    -   If you want to **skip testing and directly track with DeepSORT**, just run the cells with the pre-selected values (or change the values as you like, does not guarantee good results though) for ```best_max_age```, ```best_n_init```, ```best_min_confidence```, ```best_max_iou_distance```, ```best_max_dist```, and ```best_nms_max_overlap```, and for DeepSORT tracking, run the final cell before the tracklet visualizations.
    -   Example command that you **do not need to run manually, as the notebook builds and runs it for you**:
        ```python
        !python deep_sort_pytorch/deepsort.py --save_path "./output/test" --test_folders {test_folders_str} --appearance_model {best_model} --max_age {best_max_age} --n_init {best_n_init} --min_confidence {best_min_confidence} --max_iou_distance {best_max_iou_distance} --max_dist {best_max_dist} --nms_max_overlap {best_nms_max_overlap}
        ```
    -   After tracking, you can find the `results.txt` with the bounding boxes of the tracklets and the `log.txt` with the tracking numbers per frame in the directory:
        -   `./output/val` for **testing and comparing different configurations**
        -   `./output/test` for **runing the final tracking**
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
This project depends on source code from `SORT` for multi-object tracking (MOT), which must be cloned manually and is not included directly in the repository.
It is licensed under the GNU General Public License v3.0 (GPLv3).
See the [SORT license](https://github.com/abewley/sort/blob/master/LICENSE) for more information.

### [DeepSORT with PyTorch](https://github.com/ZQPei/deep_sort_pytorch) - MIT License
This project includes components from `deep_sort_pytorch` for appearance-based MOT.
It is licensed under the MIT License.
See the [deep_sort_pytorch license](https://github.com/JS-10/Cell_Tracking_Project_6/blob/main/deep_sort_pytorch/LICENSE) for more information.

We have made several adaptations to the original code to fit our dataset and project structure. For details on our changes, see the [adapted README](https://github.com/JS-10/Cell_Tracking_Project_6/blob/main/deep_sort_pytorch/README.md).
