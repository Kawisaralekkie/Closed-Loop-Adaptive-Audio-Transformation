#!/usr/bin/env python3
"""Setup S3 Cross-Account Replication for disaster recovery.

This script sets up replication from primary account buckets to backup account.

Usage:
    # Step 1: Run in BACKUP account to create destination buckets
    python scripts/setup_cross_account_replication.py --setup-destination
    
    # Step 2: Run in PRIMARY account to setup replication rules
    python scripts/setup_cross_account_replication.py --setup-replication
"""

import argparse
import json
import boto3

PRIMARY_ACCOUNT = "<PRIMARY_ACCOUNT_ID>"
BACKUP_ACCOUNT = "<BACKUP_ACCOUNT_ID>"
REGION = "ap-southeast-1"

# Source buckets in primary account
SOURCE_BUCKETS = [
    "<RAW_AUDIO_BUCKET>",
    "<PROCESSED_AUDIO_BUCKET>",
    "<LOGS_BUCKET>",
    "<KNOWLEDGE_BASE_BUCKET>",
]


def get_dest_bucket_name(source_bucket: str) -> str:
    """Generate destination bucket name from source."""
    # Extract the logical name part
    if "rawaudio" in source_bucket:
        return "urbansoundscape-backup-rawaudio"
    elif "processedaudio" in source_bucket:
        return "urbansoundscape-backup-processedaudio"
    elif "logs" in source_bucket:
        return "urbansoundscape-backup-logs"
    elif "knowledgebase" in source_bucket:
        return "urbansoundscape-backup-knowledgebase"
    return f"backup-{source_bucket[:50]}"


def setup_destination_buckets():
    """Create destination buckets in backup account with proper policies."""
    s3 = boto3.client("s3", region_name=REGION)
    sts = boto3.client("sts")
    
    # Verify we're in backup account
    account_id = sts.get_caller_identity()["Account"]
    if account_id != BACKUP_ACCOUNT:
        print(f"ERROR: Must run in backup account ({BACKUP_ACCOUNT}), currently in {account_id}")
        return
    
    print(f"Setting up destination buckets in backup account ({BACKUP_ACCOUNT})...")
    
    for source_bucket in SOURCE_BUCKETS:
        dest_bucket = get_dest_bucket_name(source_bucket)
        print(f"\n Creating {dest_bucket}...")
        
        try:
            # Create bucket
            s3.create_bucket(
                Bucket=dest_bucket,
                CreateBucketConfiguration={"LocationConstraint": REGION},
            )
            print(f"   Created bucket: {dest_bucket}")
        except s3.exceptions.BucketAlreadyOwnedByYou:
            print(f"   Bucket already exists: {dest_bucket}")
        except Exception as e:
            print(f"   Error creating bucket: {e}")
            continue
        
        # Enable versioning (required for replication)
        s3.put_bucket_versioning(
            Bucket=dest_bucket,
            VersioningConfiguration={"Status": "Enabled"},
        )
        print(f"   Enabled versioning")
        
        # Set bucket policy to allow replication from primary account
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "AllowReplicationFromPrimary",
                    "Effect": "Allow",
                    "Principal": {"AWS": f"arn:aws:iam::{PRIMARY_ACCOUNT}:root"},
                    "Action": [
                        "s3:ReplicateObject",
                        "s3:ReplicateDelete",
                        "s3:ReplicateTags",
                        "s3:ObjectOwnerOverrideToBucketOwner",
                    ],
                    "Resource": f"arn:aws:s3:::{dest_bucket}/*",
                },
                {
                    "Sid": "AllowReplicationVersioning",
                    "Effect": "Allow",
                    "Principal": {"AWS": f"arn:aws:iam::{PRIMARY_ACCOUNT}:root"},
                    "Action": ["s3:GetBucketVersioning", "s3:PutBucketVersioning"],
                    "Resource": f"arn:aws:s3:::{dest_bucket}",
                },
            ],
        }
        s3.put_bucket_policy(Bucket=dest_bucket, Policy=json.dumps(policy))
        print(f"   Set bucket policy for cross-account replication")
    
    print("\n✅ Destination buckets ready! Now run --setup-replication in primary account.")


def setup_replication_rules():
    """Setup replication rules in primary account."""
    s3 = boto3.client("s3", region_name=REGION)
    iam = boto3.client("iam")
    sts = boto3.client("sts")
    
    # Verify we're in primary account
    account_id = sts.get_caller_identity()["Account"]
    if account_id != PRIMARY_ACCOUNT:
        print(f"ERROR: Must run in primary account ({PRIMARY_ACCOUNT}), currently in {account_id}")
        return
    
    print(f"Setting up replication rules in primary account ({PRIMARY_ACCOUNT})...")
    
    # Create replication role
    role_name = "urbansoundscape-s3-replication-role"
    
    assume_role_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "s3.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    
    try:
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy),
            Description="Role for S3 cross-account replication",
        )
        print(f"Created IAM role: {role_name}")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"IAM role already exists: {role_name}")
    
    # Create replication policy
    replication_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetReplicationConfiguration",
                    "s3:ListBucket",
                ],
                "Resource": [f"arn:aws:s3:::{b}" for b in SOURCE_BUCKETS],
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObjectVersionForReplication",
                    "s3:GetObjectVersionAcl",
                    "s3:GetObjectVersionTagging",
                ],
                "Resource": [f"arn:aws:s3:::{b}/*" for b in SOURCE_BUCKETS],
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:ReplicateObject",
                    "s3:ReplicateDelete",
                    "s3:ReplicateTags",
                    "s3:ObjectOwnerOverrideToBucketOwner",
                ],
                "Resource": [
                    f"arn:aws:s3:::{get_dest_bucket_name(b)}/*" for b in SOURCE_BUCKETS
                ],
            },
        ],
    }
    
    try:
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName="S3ReplicationPolicy",
            PolicyDocument=json.dumps(replication_policy),
        )
        print("Attached replication policy to role")
    except Exception as e:
        print(f"Error attaching policy: {e}")
    
    role_arn = f"arn:aws:iam::{PRIMARY_ACCOUNT}:role/{role_name}"
    
    # Setup replication for each bucket
    for source_bucket in SOURCE_BUCKETS:
        dest_bucket = get_dest_bucket_name(source_bucket)
        print(f"\n Setting up replication: {source_bucket} -> {dest_bucket}")
        
        # Ensure source bucket has versioning enabled
        try:
            s3.put_bucket_versioning(
                Bucket=source_bucket,
                VersioningConfiguration={"Status": "Enabled"},
            )
            print(f"   Enabled versioning on source")
        except Exception as e:
            print(f"   Warning: Could not enable versioning: {e}")
        
        # Setup replication configuration
        replication_config = {
            "Role": role_arn,
            "Rules": [
                {
                    "ID": "CrossAccountReplication",
                    "Status": "Enabled",
                    "Priority": 1,
                    "Filter": {"Prefix": ""},
                    "Destination": {
                        "Bucket": f"arn:aws:s3:::{dest_bucket}",
                        "Account": BACKUP_ACCOUNT,
                        "AccessControlTranslation": {"Owner": "Destination"},
                    },
                    "DeleteMarkerReplication": {"Status": "Enabled"},
                }
            ],
        }
        
        try:
            s3.put_bucket_replication(
                Bucket=source_bucket,
                ReplicationConfiguration=replication_config,
            )
            print(f"   ✅ Replication configured")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n✅ Replication setup complete!")
    print("New objects will be automatically replicated to backup account.")
    print("To replicate existing objects, use S3 Batch Replication.")


def check_replication_status():
    """Check replication status for all buckets."""
    s3 = boto3.client("s3", region_name=REGION)
    
    print("Checking replication status...\n")
    
    for source_bucket in SOURCE_BUCKETS:
        dest_bucket = get_dest_bucket_name(source_bucket)
        print(f"{source_bucket}:")
        
        try:
            config = s3.get_bucket_replication(Bucket=source_bucket)
            rules = config.get("ReplicationConfiguration", {}).get("Rules", [])
            for rule in rules:
                status = rule.get("Status", "Unknown")
                dest = rule.get("Destination", {}).get("Bucket", "Unknown")
                print(f"   Status: {status}")
                print(f"   Destination: {dest}")
        except s3.exceptions.ClientError as e:
            if "ReplicationConfigurationNotFoundError" in str(e):
                print("   No replication configured")
            else:
                print(f"   Error: {e}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Setup S3 cross-account replication")
    parser.add_argument(
        "--setup-destination",
        action="store_true",
        help="Create destination buckets in backup account",
    )
    parser.add_argument(
        "--setup-replication",
        action="store_true",
        help="Setup replication rules in primary account",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check replication status",
    )
    
    args = parser.parse_args()
    
    if args.setup_destination:
        setup_destination_buckets()
    elif args.setup_replication:
        setup_replication_rules()
    elif args.status:
        check_replication_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
