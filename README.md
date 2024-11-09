# Project Setup and Execution Guide

This guide provides instructions on setting up the project environment, preparing data, running specific notebooks, and configuring GPU clusters for optimal performance.

---

## 1. Data Preparation

1. **Download Data**: Download the processed benchmark data from the following URL:
   ```
   ![Download Data](https://drive.google.com/drive/folders/1hBkz8CGcRTdsb8LcTi56v75a_xGI1pLH?usp=sharing)
   ```
   
2. **Upload Data**: Upload the downloaded data to Google Drive in a folder named `Benchmark Data`.

---

## 2. Running Notebooks for Embedding Generation

To work with different algorithms, run the appropriate notebook as described below:

- **Algorithm 1**: Run `1_Embedding_without_Prompt.ipynb`.
- **Algorithm 2**: Run `2_Embedding_Prompt_for_each_datapoint.ipynb`.
- **Algorithm 3**: Run `3_Embedding_Prompt_for_each_row.ipynb`.

Each notebook will use `C_train`, `C_test`, and `C_val` `.npy` files to create new embedded files with the same names.

### Concatenate Embedded Data

1. To concatenate `C_train` embeddings with `N_train`, run the following notebook:
   ```
   4_Combine_C_train_and_N_train_and_Generate_Toml_Algo3.ipynb
   ```

---

## 3. GPU Cluster Setup

### Step 1: Directory and Permissions

1. Create a directory and set permissions:
   ```bash
   mkdir my_model
   sudo chmod -R 777 my_model
   ```

2. Clone the model repository:
   ```bash
   cd my_model
   git clone https://github.com/jeevitharodrigues/Categorical_Data_Encoding_via_LLMs.git
   ```

3. Set the project directory path:
   ```bash
   export PROJECT_DIR=/home/ubuntu/my_model/Categorical_Data_Encoding_via_LLMs/base_models
   ```

### Step 2: NVIDIA Driver Installation

1. Update the system and install the NVIDIA driver:
   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-driver-470
   sudo reboot
   ```

2. Verify the GPU installation with:
   ```bash
   nvidia-smi
   ```

   Expected output:
   ```
   +-----------------------------------------------------------------------------+
   | NVIDIA-SMI 470.256.02   Driver Version: 470.256.02   CUDA Version: 11.4     |
   +-----------------------------------------------------------------------------+
   ```

   If the above output isn’t received, run:
   ```bash
   sudo apt install nvidia-cuda-toolkit
   sudo apt-get upgrade
   nvidia-smi
   ```

---

## 4. Environment Setup with Miniconda and Python 3.8

1. Create directories and install Miniconda:
   ```bash
   mkdir my_pack
   cd my_pack
   mkdir -p ~/miniconda3
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
   bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
   rm ~/miniconda3/miniconda.sh
   ```

2. Install Python 3.8:
   ```bash
   sudo wget https://www.python.org/ftp/python/3.8.8/Python-3.8.8.tgz
   sudo tar xfv Python-3.8.8.tgz
   cd Python-3.8.8
   sudo ./configure --enable-optimizations
   sudo make altinstall
   python3.8 -V  # Should output Python 3.8.8
   ```

---

## 5. Set Up Conda Environment and Install Dependencies

1. Set the project path:
   ```bash
   cd /home/ubuntu/my_model/Categorical_Data_Encoding_via_LLMs/base_models
   export PATH=~/miniconda3/bin:$PATH
   ```

2. Create and activate the environment:
   ```bash
   conda create -n base_models python=3.8.8 -y
   conda activate base_models
   ```

3. Install dependencies:
   ```bash
   conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1.243 numpy=1.19.2 -c pytorch -y
   conda install cudnn=7.6.5 -c anaconda -y
   pip install -r requirements.txt
   conda install nodejs -y
   jupyter labextension install @jupyter-widgets/jupyterlab-manager
   ```

4. Set environment variables:
   ```bash
   conda env config vars set PYTHONPATH=${PYTHONPATH}:${PROJECT_DIR}
   conda env config vars set PROJECT_DIR=${PROJECT_DIR}
   conda env config vars set LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
   conda env config vars set CUDA_HOME=${CONDA_PREFIX}
   conda env config vars set CUDA_ROOT=${CONDA_PREFIX}
   ```

   Reload the environment:
   ```bash
   conda deactivate
   conda activate base_models
   ```

---

## 6. Final Steps

1. **Upload Concatenated Data**: Place the concatenated data into the `data` folder.

2. **Run Model**: To run any model, use the following command:
   ```bash
   python3.8 bin/{model_name}.py path_to_toml -f
   ```

   Example:
   ```bash
   python3.8 bin/resnet.py output/adult/resnet/tuned/0.toml -f
   ```