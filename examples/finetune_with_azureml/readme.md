# Instructions to run full fine-tuning on the libero dataset 

## Required resources: 
- Azure ML workspace. In the following, the workspace is called `<ws_name>` and it is in the resource group `<rg_name>`.
    - Users need "AzureML Data Scientist" role
- Azure ML GPU compute instance. In the following, it is called `<gpu_vm_name>`.
    - With at least one A100 GPU with 80GB of memory
    - With a system/user assigned managed identity
- Azure ML CPU compute instance. In the following, it is called `<cpu_vm_name>`.
    - With at least 64 GB RAM. 
    - With a system/user assigned managed identity.
- ADLS Gen2 storage account with hierarchical namespace enabled. In the following examples, the storage account is called `<sa_name>` and container is called `<container_name>`. 
    - Users and VM identities need **"Storage blob data contributor"** role
    - The `<container_name>` container in the `<sa_name>` storage account needs to be added as an Azure ML datastore (make sure to mount it as ADLS Gen2 with HNS otherwise you'll face `Os.Errors` during checkpoint writing!). The datastore is called `<ds_name>`.

## Create the AzureML environment

First, create the AzureML environment. Here and in the following steps it's called it `<env_name>`. Environments have versions referred to as `<env_version>`.
```
 az ml environment create --name <env_name> --build-context . --dockerfile-path examples/finetune_with_azureml/Dockerfile --resource-group  <rg_name> --workspace-name <ws_name>
```

## Download the libero dataset and generate norm stats

Next, download the "libero" dataset and upload it to the `<container_name>` container in the `<sa_name>` storage account. Once that is done, generate the norm stats of the dataset and upload them to the same location.

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
* Use git-lfs to download the dataset (preferably in the /mnt directory of your VM, to avoid space issues):
    
    ```
    (optional) sudo mkdir -p /mnt/data && cd /mnt/data
    git lfs install
    git clone https://huggingface.co/datasets/physical-intelligence/libero
    ```

* Copy the libero dataset to the `<container_name>` container in the `<sa_name>` storage account under the path "pi0_fast_libero/physical-intelligence/libero".
   Login with azcopy first. If you are using a VM with managed identity you can use:
   ```
   azcopy login --identity
   ```
   Copy the files
   ```
   azcopy copy "/path/to/downloaded/libero/dataset/*" "https://<sa_name>.dfs.core.windows.net/<container_name>/pi0_fast_libero/physical-intelligence/libero/" --recursive=true
   ```

### Generate the norm stats

Calculate the norm stats of the libero dataset.

> Note: This is done in an AzureML job that will write the generated "norm_stats.json" file in the same directory in the `<container_name>` container where we previously copied the libero dataset. 
  This step cannot run on a VM with GPU, so we use our CPU VM. 
  The generation of the norm stats can also be done locally instead of in an AzureML job, then you can copy the generated "norm_stats.json" file manually into the correct location in the `<container_name>` container using azcopy as we did in the previous step. 

> Note: Instead of using the `--set` options, you can directly substitute the placeholder values in the yaml file before running.

```
 az ml job create --file examples/finetune_with_azureml/data-norm-stats.yaml --resource-group <rg_name> --workspace-name <ws_name> --set compute=<cpu_vm_name> --set environment="azureml:<env_name>:<env_version>" --set outputs.ASSETS_BASE.path="azureml://datastores/<ds_name>/paths/pi0_fast_libero/physical-intelligence/libero/" --set outputs.HF_LEROBOT_HOME.path="azureml://datastores/<ds_name>/paths/pi0_fast_libero/"
```

The data should be ready at this point. Your container structure should resemble:

![alt text](images/container_structure.png)

## Run the fine-tuning

Finally, run the fine-tuning. Select your own experiment name and substitute all placeholder values.

The fine-tuning will run for 30,000 steps with batches of size 32 by default. Checkpoints will be generated every 1000 steps. These configs (and others) can be modified in the `train.yaml` file.

> Note: Instead of using the `--set` options, you can directly substitute the placeholder values in the yaml file before running.

```
az ml job create --file examples/finetune_with_azureml/train.yaml --resource-group <rg_name> --workspace-name <ws_name> --set inputs.pi0_experiment_name="my_experiment" --set compute="<gpu_vm_name>" --set environment="azureml:<env_name>:<env_version>" --set outputs.ASSETS_BASE.path="azureml://datastores/<ds_name>/paths/" --set outputs.HF_LEROBOT_HOME.path="azureml://datastores/<ds_name>/paths/pi0_fast_libero/" --set outputs.MODEL_CHECKPOINTS.path="azureml://datastores/<ds_name>/paths/finetuned_model/"
```

To overwrite the checkpoint dir if it exists: 
```
az ml job create --file examples/finetune_with_azureml/train.yaml --resource-group <rg_name> --workspace-name <ws_name> --set inputs.pi0_experiment_name="my_experiment" --set compute="<gpu_vm_name>" --set environment="azureml:<env_name>:<env_version>" --set outputs.ASSETS_BASE.path="azureml://datastores/<ds_name>/paths/" --set outputs.HF_LEROBOT_HOME.path="azureml://datastores/<ds_name>/paths/pi0_fast_libero/" --set outputs.MODEL_CHECKPOINTS.path="azureml://datastores/<ds_name>/paths/finetuned_model/" --set inputs.extra_flags="--overwrite"
```

To continue from last checkpoint: 
```
az ml job create --file examples/finetune_with_azureml/train.yaml --resource-group <rg_name> --workspace-name <ws_name> --set inputs.experiment_name="my_experiment" --set compute="<gpu_vm_name>" --set environment="azureml:<env_name>:<env_version>" --set outputs.ASSETS_BASE.path="azureml://datastores/<ds_name>/paths/" --set outputs.HF_LEROBOT_HOME.path="azureml://datastores/<ds_name>/paths/pi0_fast_libero/" --set outputs.MODEL_CHECKPOINTS.path="azureml://datastores/<ds_name>/paths/finetuned_model/" --set inputs.extra_flags="--resume"
```