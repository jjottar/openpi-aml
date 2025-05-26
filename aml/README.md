## Pre-processing the LIBERO dataset for fine-tuning

Instructions below assume an Azure ML compute instance is used for the pre-processing. To avoid running out of space, choose an instance with a large storage, such as the memory-optimized [Standard_D15_v2 (20 cores, 140 GB RAM, 1000 GB disk)](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/memory-optimized/dv2-dsv2-series-memory), for example.

* If not already done, set the git config on your VM:

    ```
    git config --global user.name "Your name"
    git config --global user.email "Your email@address"
    ```

* Install git-lfs on your VM:

    ```
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs
    ```
* Use git-lfs to download the dataset (preferrably in the /mnt directory of your VM, to avoid space issues):
    
    ```
    (optional) sudo mkdir -p /mnt/data && cd /mnt/data
    git lfs install
    git clone https://huggingface.co/datasets/openvla/modified_libero_rlds
    ```
* As explained in [examples/libero/convert_libero_data_to_lerobot.py](../examples/libero/convert_libero_data_to_lerobot.py), we now have to run a script to transform the dataset to LeRobot format. This script requires tensforflow_datasets, so install it with

    ```
    uv pip install tensorflow tensorflow_datasets
    ```

    Then change the environment variable which defines the directory where the processed data will be written

    ```
    export LEROBOT_HOME=<output_data_dir>
    ```


    **Note:** you may have to give elevated permissions on this directory, with `sudo chown -R $USER:$USER <output_data_dir>`. Further, if you want to modify the specific subfolder where the processed output data will be written, modify the variable `REPO_NAME` in [examples/libero/convert_libero_data_to_lerobot.py](../examples/libero/convert_libero_data_to_lerobot.py). 

    Finally, run 

    ```
    uv run convert_libero_data_to_lerobot.py --data_dir <raw_data_dir>
    ```

    where `<raw_data_dir>` is the directory containing the raw dataset downloaded from Hugging face (e.g. `mnt/data/modified_libero_rlds`).
