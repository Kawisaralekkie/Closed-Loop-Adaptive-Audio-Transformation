#!/usr/bin/env python3
"""Export all Urban Soundscape data from S3 to local or another account.

Usage:
    # Export to local directory
    python scripts/backup_export.py --output-dir ./backup

    # Export to another S3 bucket (cross-account)
    python scripts/backup_export.py --dest-bucket my-backup-bucket --dest-profile backup-account

    # Export specific data types only
    python scripts/backup_export.py --output-dir ./backup --types raw-audio,logs
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


# S3 bucket prefixes to export
BUCKET_TYPES = {
    "raw-audio": {"env": "RAW_AUDIO_BUCKET", "prefix": ""},
    "processed-audio": {"env": "PROCESSED_AUDIO_BUCKET", "prefix": ""},
    "knowledge-base": {"env": "KB_S3_BUCKET", "prefix": "kb/"},
    "logs": {"env": "LOGS_BUCKET", "prefix": ""},
}


def get_bucket_name(bucket_type: str) -> str | None:
    """Get bucket name from environment or CloudFormation outputs."""
    env_key = BUCKET_TYPES[bucket_type]["env"]
    name = os.environ.get(env_key)
    if name:
        return name

    # Try to get from CDK outputs
    try:
        result = subprocess.run(
            ["aws", "cloudformation", "describe-stacks",
             "--stack-name", "UrbanSoundscape-Storage",
             "--query", "Stacks[0].Outputs"],
            capture_output=True, text=True, check=True,
        )
        outputs = json.loads(result.stdout) or []
        for out in outputs:
            if bucket_type.replace("-", "") in out.get("OutputKey", "").lower():
                return out["OutputValue"]
    except Exception:
        pass
    return None


def export_to_local(bucket_name: str, prefix: str, output_dir: str, label: str) -> bool:
    """Sync S3 bucket to local directory."""
    dest = os.path.join(output_dir, label)
    os.makedirs(dest, exist_ok=True)
    s3_path = f"s3://{bucket_name}/{prefix}" if prefix else f"s3://{bucket_name}"
    print(f"  Syncing {s3_path} → {dest}")
    result = subprocess.run(
        ["aws", "s3", "sync", s3_path, dest, "--no-progress"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        return False
    return True


def export_to_s3(
    src_bucket: str, prefix: str, dest_bucket: str,
    label: str, profile: str | None = None,
) -> bool:
    """Copy S3 bucket to another S3 bucket (cross-account)."""
    src = f"s3://{src_bucket}/{prefix}" if prefix else f"s3://{src_bucket}"
    dest = f"s3://{dest_bucket}/backup/{label}/"
    cmd = ["aws", "s3", "sync", src, dest, "--no-progress"]
    if profile:
        cmd.extend(["--profile", profile])
    print(f"  Syncing {src} → {dest}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        return False
    return True


def create_manifest(output_dir: str, results: dict) -> str:
    """Create a backup manifest file."""
    manifest = {
        "backup_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_account": os.environ.get("AWS_ACCOUNT_ID", "unknown"),
        "source_region": os.environ.get("AWS_DEFAULT_REGION", "ap-southeast-1"),
        "buckets": results,
        "restore_instructions": {
            "step_1": "Create new S3 buckets in target account",
            "step_2": "aws s3 sync ./backup/<type> s3://<new-bucket>/",
            "step_3": "Update environment variables with new bucket names",
            "step_4": "Deploy CDK stack: cd infra && cdk deploy --all",
        },
    }
    path = os.path.join(output_dir, "backup_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved → {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Export Urban Soundscape data")
    parser.add_argument("--output-dir", help="Local directory for backup")
    parser.add_argument("--dest-bucket", help="Destination S3 bucket (cross-account)")
    parser.add_argument("--dest-profile", help="AWS CLI profile for destination account")
    parser.add_argument(
        "--types",
        default="raw-audio,processed-audio,knowledge-base,logs",
        help="Comma-separated data types to export",
    )
    args = parser.parse_args()

    if not args.output_dir and not args.dest_bucket:
        parser.error("Specify --output-dir (local) or --dest-bucket (S3)")

    types = [t.strip() for t in args.types.split(",")]
    results = {}

    print(f"Starting backup: {', '.join(types)}")
    print("=" * 50)

    for btype in types:
        if btype not in BUCKET_TYPES:
            print(f"  SKIP: unknown type '{btype}'")
            continue

        bucket_name = get_bucket_name(btype)
        if not bucket_name:
            print(f"  SKIP: {btype} — bucket name not found")
            results[btype] = {"status": "skipped", "reason": "bucket not found"}
            continue

        prefix = BUCKET_TYPES[btype]["prefix"]
        print(f"\n[{btype}] bucket={bucket_name}")

        if args.output_dir:
            ok = export_to_local(bucket_name, prefix, args.output_dir, btype)
        else:
            ok = export_to_s3(
                bucket_name, prefix, args.dest_bucket, btype, args.dest_profile,
            )

        results[btype] = {
            "status": "success" if ok else "failed",
            "bucket": bucket_name,
        }

    if args.output_dir:
        create_manifest(args.output_dir, results)

    # Summary
    success = sum(1 for r in results.values() if r.get("status") == "success")
    total = len(results)
    print(f"\nDone: {success}/{total} exported successfully")


if __name__ == "__main__":
    main()
