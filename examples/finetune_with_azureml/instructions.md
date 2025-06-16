# Normalize data
# Start finetuning
az ml job create --file examples/finetune_with_azureml/train.yaml \
--resource-group robotics-ch-north-secure \
--workspace-name robotics-ch-north-secure \
--set inputs.pi0_experiment_name="openpi0fast_50_episodes_PI_impl" \
--set compute="nc80adis-h100-v5-single" \
--set environment="azureml:openpi0-env:1" \
--set outputs.ASSETS_BASE.path="azureml://datastores/nora_datasets/paths/" \
--set outputs.HF_LEROBOT_HOME.path="azureml://datastores/nora_datasets/paths/pi0_fast_custom/" \
--set outputs.MODEL_CHECKPOINTS.path="azureml://datastores/nora_datasets/paths/finetuned_model/" \
--set inputs.extra_flags="--data.repo_id noraabk/so101-goat-picking-v1 --overwrite"
