#!/usr/bin/env python3
"""Re-run failed Step Function executions.

Finds all FAILED executions, extracts the S3 key from each,
and triggers new executions for those files.

Usage:
    python3 scripts/rerun_failed_step_functions.py [--batch-size 3] [--delay 360] [--dry-run]
    python3 scripts/rerun_failed_step_functions.py --since today
    python3 scripts/rerun_failed_step_functions.py --since 2026-03-27
"""

import argparse
import json
import os
import time
import uuid
from datetime import datetime, timezone, timedelta

import boto3

# AWS identifiers come from environment variables (no account-specific values
# are committed). Set: AWS_ACCOUNT_ID, AWS_REGION, RAW_AUDIO_BUCKET.
AWS_ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID", "<AWS_ACCOUNT_ID>")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
STATE_MACHINE_ARN = os.environ.get(
    "STATE_MACHINE_ARN",
    f"arn:aws:states:{AWS_REGION}:{AWS_ACCOUNT_ID}:stateMachine:urban-soundscape-pipeline")
RAW_BUCKET = os.environ.get("RAW_AUDIO_BUCKET", "<RAW_AUDIO_BUCKET>")


def parse_since(since_str: str) -> datetime:
    """Parse --since argument to datetime."""
    if since_str.lower() == "today":
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        return today
    elif since_str.lower() == "yesterday":
        yesterday = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        return yesterday
    else:
        # Parse as date string (YYYY-MM-DD)
        dt = datetime.strptime(since_str, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)


def get_failed_executions(sfn_client, since: datetime | None = None) -> list[dict]:
    """Get all failed executions with their input S3 keys."""
    failed = []
    paginator = sfn_client.get_paginator("list_executions")
    
    for page in paginator.paginate(
        stateMachineArn=STATE_MACHINE_ARN,
        statusFilter="FAILED",
    ):
        for execution in page["executions"]:
            start_date = execution["startDate"]
            
            # Filter by since date if provided
            if since and start_date.replace(tzinfo=timezone.utc) < since:
                continue
            
            # Get execution details to extract input
            details = sfn_client.describe_execution(
                executionArn=execution["executionArn"]
            )
            try:
                input_data = json.loads(details.get("input", "{}"))
                # Support both old format (s3_key) and new format (detail.object.key)
                s3_key = input_data.get("s3_key") or input_data.get("detail", {}).get("object", {}).get("key")
                if s3_key:
                    failed.append({
                        "execution_arn": execution["executionArn"],
                        "name": execution["name"],
                        "s3_key": s3_key,
                        "start_date": start_date,
                    })
            except json.JSONDecodeError:
                print(f"Warning: Could not parse input for {execution['name']}")
    
    return failed


def trigger_execution(sfn_client, s3_bucket: str, s3_key: str, run_mode: str = "all") -> str:
    """Trigger a new Step Function execution."""
    execution_name = f"rerun-{s3_key.replace('/', '-').replace('.wav', '')}-{uuid.uuid4().hex[:8]}"
    
    response = sfn_client.start_execution(
        stateMachineArn=STATE_MACHINE_ARN,
        name=execution_name[:80],
        input=json.dumps({
            "detail": {
                "bucket": {"name": s3_bucket},
                "object": {"key": s3_key},
            },
            "run_mode": run_mode,
        }),
    )
    return response["executionArn"]


def main():
    parser = argparse.ArgumentParser(description="Re-run failed Step Function executions")
    parser.add_argument("--batch-size", type=int, default=3, help="Number of concurrent executions")
    parser.add_argument("--delay", type=int, default=360, help="Delay between batches (seconds)")
    parser.add_argument("--dry-run", action="store_true", help="List failed executions without re-running")
    parser.add_argument("--since", type=str, help="Only re-run failures since date (today, yesterday, or YYYY-MM-DD)")
    parser.add_argument("--run-mode", default="all",
                        choices=["all", "fixed", "rule_based", "llm_no_memory", "llm_with_memory"],
                        help="Pipeline mode for re-runs (default: all)")
    args = parser.parse_args()

    sfn_client = boto3.client("stepfunctions", region_name="ap-southeast-1")
    
    since = parse_since(args.since) if args.since else None
    since_str = f" since {args.since}" if args.since else ""
    
    print(f"Fetching failed executions{since_str}...")
    failed = get_failed_executions(sfn_client, since=since)
    
    if not failed:
        print("No failed executions found.")
        return
    
    print(f"\nFound {len(failed)} failed executions:")
    for f in failed:
        print(f"  - {f['s3_key']} (started: {f['start_date']})")
    
    if args.dry_run:
        print("\n[DRY RUN] No executions triggered.")
        return
    
    print(f"\nRe-running with batch_size={args.batch_size}, delay={args.delay}s...")
    
    # Get unique S3 keys (avoid duplicates)
    unique_keys = list({f["s3_key"] for f in failed})
    print(f"Unique files to re-run: {len(unique_keys)}")
    
    for i, s3_key in enumerate(unique_keys):
        arn = trigger_execution(sfn_client, RAW_BUCKET, s3_key, args.run_mode)
        print(f"[{i+1}/{len(unique_keys)}] Triggered: {s3_key}")
        
        # Delay between batches
        if (i + 1) % args.batch_size == 0 and i + 1 < len(unique_keys):
            print(f"Waiting {args.delay}s before next batch...")
            time.sleep(args.delay)
    
    print(f"\nDone! Triggered {len(unique_keys)} executions.")


if __name__ == "__main__":
    main()
