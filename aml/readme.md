 az ml data create --file aml/dataset.yaml --resource-group robotics-ch-north --workspace-name robotics-ch-north

Allow storage account key access

 az ml environment create --name libero-env --build-context scripts/docker --dockerfile-path serve_policy.Dockerfile --resource-group  robotics-ch-north --workspace-name robotics-ch-north --datastore libero

Configure the default storage to allow access from all networks, and allow key access and give myself permission cause I cant change the datastore for the jobs :(

 az ml job create --file aml/data-norm-stats.yaml --resource-group robotics-ch-north --workspace-name robotics-ch-north