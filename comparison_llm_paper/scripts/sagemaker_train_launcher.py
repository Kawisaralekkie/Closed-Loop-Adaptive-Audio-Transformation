#!/usr/bin/env python3
"""Launch YAMNet transfer-learning as a SageMaker Training Job.

Submits ``scripts/train_yamnet_transfer.py`` to a managed SageMaker training
container. SageMaker provisions the instance, runs the script, uploads the
model artifacts to S3, then tears the instance down automatically (pay only
for training time).

Prerequisites:
  - Run from a machine/notebook with an IAM role that can create SageMaker
    Training Jobs and read/write the S3 buckets.
  - The split CSVs + label_map already uploaded to S3 (run
    prepare_train_test_split.py first).

Usage:
    python3 scripts/sagemaker_train_launcher.py \
        --role-arn arn:aws:iam::<AWS_ACCOUNT_ID>:role/SageMakerExecutionRole \
        --bucket <RAW_AUDIO_BUCKET> \
        --meta-prefix cityspeechmix/cityspeechmixed_meta \
        --model-prefix cityspeechmix/models/yamnet_transfer \
        --instance-type ml.m5.xlarge
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch YAMNet transfer SageMaker training job")
    ap.add_argument("--role-arn", required=True, help="SageMaker execution IAM role ARN")
    ap.add_argument("--bucket", required=True, help="S3 bucket with audio + metadata")
    ap.add_argument("--meta-prefix", default="cityspeechmix/cityspeechmixed_meta",
                    help="Prefix holding train_split.csv / test_split.csv / label_map.json")
    ap.add_argument("--model-prefix", default="cityspeechmix/models/yamnet_transfer",
                    help="Prefix to upload trained model artifacts")
    ap.add_argument("--region", default="ap-southeast-1")
    ap.add_argument("--instance-type", default="ml.m5.xlarge",
                    help="ml.m5.xlarge (CPU) is plenty for this small dataset; "
                         "use ml.g4dn.xlarge for GPU")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-run-seconds", type=int, default=3600)
    args = ap.parse_args()

    import boto3
    import sagemaker
    from sagemaker.tensorflow import TensorFlow

    # Bind the SageMaker session to the requested region (otherwise it silently
    # falls back to the AWS SDK default region, which may not be where the
    # bucket / quota live).
    boto_sess = boto3.Session(region_name=args.region)
    sess = sagemaker.Session(boto_session=boto_sess)
    meta = f"s3://{args.bucket}/{args.meta_prefix.rstrip('/')}"
    model_out = f"s3://{args.bucket}/{args.model_prefix.rstrip('/')}"

    # SageMaker TensorFlow estimator runs our training script in a managed
    # TF container. Extra deps (tensorflow_hub, soundfile, scikit-learn) are
    # installed automatically from scripts/requirements.txt because it lives in
    # source_dir alongside the entry point.
    hyperparameters = {
        "train-csv": f"{meta}/train_split.csv",
        "test-csv": f"{meta}/test_split.csv",
        "label-map": f"{meta}/label_map.json",
        "audio-bucket": args.bucket,
        "out-dir": "/opt/ml/model",          # SageMaker auto-uploads this to S3
        "epochs": args.epochs,
        "batch-size": args.batch_size,
    }

    estimator = TensorFlow(
        entry_point="train_yamnet_transfer.py",
        source_dir="scripts",                 # ships scripts/ (+ requirements.txt)
        role=args.role_arn,
        instance_count=1,
        instance_type=args.instance_type,
        framework_version="2.13",
        py_version="py310",
        hyperparameters=hyperparameters,
        max_run=args.max_run_seconds,
        output_path=model_out,                # final model.tar.gz lands here
        base_job_name="yamnet-transfer-csm",
        sagemaker_session=sess,
    )

    job_name = f"yamnet-transfer-csm-{int(time.time())}"
    print(f"Submitting SageMaker training job: {job_name}")
    print(f"  instance: {args.instance_type}")
    print(f"  meta:     {meta}")
    print(f"  output:   {model_out}")
    estimator.fit(job_name=job_name, wait=True, logs="All")

    print("\nTraining job complete.")
    print(f"Model artifacts: {estimator.model_data}")


if __name__ == "__main__":
    main()
