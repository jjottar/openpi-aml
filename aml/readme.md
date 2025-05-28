 az ml data create --file aml/dataset.yaml --resource-group robotics-ch-north --workspace-name robotics-ch-north

Allow storage account key access

 az ml environment create --name libero-env --build-context . --dockerfile-path scripts/docker/serve_policy.Dockerfile --resource-group  robotics-ch-north --workspace-name robotics-ch-north

Configure the default storage to allow access from all networks, and allow key access and give myself permission cause I cant change the datastore for the jobs :(
I have to add LEROBOT_HOME, otherwise it will redownload the data. Also, it has to be an output because an input doesnt get substituted correctly.
Dont use a GPU machine.

 az ml job create --file aml/data-norm-stats.yaml --resource-group robotics-ch-north --workspace-name robotics-ch-north

 az ml job create --file aml/train.yaml --resource-group robotics-ch-north --workspace-name robotics-ch-north