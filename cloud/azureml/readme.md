# Instructions to run full fine-tuning on the libero dataset 

## Required resources: 
- Azure ML workspace. In the following examples and AzureML yaml scripts, the workspace is called "robotics-ch-north" and it is in the resource group "robotics-ch-north".
    - Users need "AzureML Data Scientist" role
- Azure ML GPU compute instance. In the following examples and AzureML yaml scripts it is called "vm-nc96ads-a100".
    - With at least one A100 GPU with 80GB of memory
    - With a system/user assigned managed identity
- Azure ML CPU compute instance. In the following example and AzureML yaml scripts it is called "vm-d48a-v4".
    - With at least 64 GB RAM. 
    - Wtih a system/user assigned managed identity.
- ADLS Gen2 storage account with hierarchical namespace enabled. In the following examples and AzureML yaml scripts, the storage account and container are both called "libero". 
    - Users and VM identities need "Storage blob data contributor" role
    - The "libero" container in the "libero" storage account needs to be added as an Azure ML datastore (make sure to mount it as ADLS Gen2 with HNS otherwise you'll face `Os.Errors` during checkpoint writing!)

## Create the AzureML environment

First, create the AzureML environment. Here and in the following steps it's called it `libero-env`.
```
 az ml environment create --name libero-env --build-context . --dockerfile-path scripts/docker/serve_policy.Dockerfile --resource-group  robotics-ch-north --workspace-name robotics-ch-north
```

## Download the libero dataset and generate norm stats

Next, download the "libero" dataset and upload it to the "libero" container in the "libero" storage account. Once that is done, generate the norm stats of the dataset and upload them to the same location.

### Download the libero dataset

> Note: Instructions assume an Azure ML compute instance is used for downloading the dataset. To avoid running out of space, choose an instance with a large storage, such as the memory-optimized [Standard_D15_v2 (20 cores, 140 GB RAM, 1000 GB disk)](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/memory-optimized/dv2-dsv2-series-memory). You could also run these steps locally or in an AzureML job.

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
    git clone https://huggingface.co/datasets/physical-intelligence/libero
    ```

* Copy the libero dataset to the "libero" container in the "libero" storage account under the path "pi0_fast_libero/physical-intelligence/libero".
   Login with azcopy first. If you are using a VM with managed identity you can use:
   ```
   azcopy login --identity
   ```
   Copy the files
   ```
   azcopy "/path/to/downloaded/libero/dataset/*" "https://libero.blob.core.windows.net/libero/pi0_fast_libero/physical-intelligence/libero/" --recursive=true
   ```

### Generate the norm stats

Calculate the norm stats of the libero dataset.

> Note: This is done in an AzureML job that will write the generated "norm_stats.json" file in the same directory in the "libero" container where we previously copied the "libero" dataset. 
  This step cannot run on a VM with GPU, so we use our CPU VM. 
  The generation of the norm stats can also be done locally instead of in an AzureML job, then you can copy the generated "norm_stats.json" file manually into the correct location in the "libero" container using azcopy as we did in the previous step. 

> Note: The environment version may need to be updated in the `data-norm-stats.yaml` file. You can use `--set environment` to change the value or edit the file directly. 

```
 az ml job create --file cloud/azureml/data-norm-stats.yaml --resource-group robotics-ch-north --workspace-name robotics-ch-north --set compute="vm-d48a-v4"
```

The data should be ready at this point. Your container structure should resemble:

![alt text](images/container_structure.png)

## Run the fine-tuning

Finally, run the fine-tuning. Select your own experiment name.

The fine-tuning will run for 30,000 steps with batches of size 32 by default. Checkpoints will be generated every 1000 steps. These configs (and others) can be modified in the `train.yaml` file.

> Note: The environment version may need to be updated in the `train.yaml` file. You can use `--set environment` to change the value or edit the file directly. 

```
az ml job create --file cloud/azureml/train.yaml --resource-group robotics-ch-north --workspace-name robotics-ch-north --set inputs.experiment_name="my_experiment" --set compute="vm-nc96ads-a100" 
```

To overwrite the checkpoint dir if it exists: 
```
az ml job create --file cloud/azureml/train.yaml --resource-group robotics-ch-north --workspace-name robotics-ch-north --set inputs.experiment_name="my_experiment" --set compute="vm-nc96ads-a100" --set inputs.extra_flags="--overwrite"
```

To continue from last checkpoint: 
```
az ml job create --file cloud/azureml/train.yaml --resource-group robotics-ch-north --workspace-name robotics-ch-north --set inputs.experiment_name="my_experiment" --set compute="vm-nc96ads-a100" --set inputs.extra_flags="--resume"
```