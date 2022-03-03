v=$DATABASE_URL_NO_PARAMS
v2=${v::-7}


if [$CLOUD_PROVIDER == "AWS"]
then
    export MLFLOW_BUCKET=s3://$HYPERPLANE_AWS_BUCKET/user-mlflow
fi

if [$CLOUD_PROVIDER == "GCP"]
then
    export MLFLOW_BUCKET=gs://$HYPERPLANE_GCP_BUCKET/user-mlflow
fi

mlflow server --backend-store-uri $v2/$HYPERPLANE_JOB_PARAMETER_USERDIR --default-artifact-root $MLFLOW_BUCKET/$HYPERPLANE_JOB_PARAMETER_USERDIR --host 0.0.0.0 --port 8787