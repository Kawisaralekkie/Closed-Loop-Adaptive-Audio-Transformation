#!/usr/bin/env python3
"""Download logs from CloudWatch and S3 for Step Function executions.

Usage:
    # Download Step Function CloudWatch logs (last 24 hours)
    python3 scripts/download_logs.py --source cloudwatch --hours 24

    # Download S3 pipeline reports (Fixed + Agentic)
    python3 scripts/download_logs.py --source s3

    # Download both
    python3 scripts/download_logs.py --source all --hours 48

    # Download logs for today only
    python3 scripts/download_logs.py --source all --since today
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import boto3

# Configuration
REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
LOG_GROUP_NAME = os.environ.get(
    "LOG_GROUP_NAME", "/aws/vendedlogs/states/urban-soundscape-pipeline")
LOGS_BUCKET = os.environ.get("LOGS_BUCKET", "<LOGS_BUCKET>")
OUTPUT_DIR = "logs"


def parse_since(since_str: str) -> datetime:
    """Parse --since argument to datetime."""
    if since_str.lower() == "today":
        return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    elif since_str.lower() == "yesterday":
        return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    else:
        dt = datetime.strptime(since_str, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)


def download_cloudwatch_logs(hours: int = 24, since: datetime | None = None, output_dir: str = OUTPUT_DIR):
    """Download Step Function logs from CloudWatch."""
    logs_client = boto3.client("logs", region_name=REGION)
    
    # Calculate time range
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    if since:
        start_time = int(since.timestamp() * 1000)
    else:
        start_time = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp() * 1000)
    
    print(f"Downloading CloudWatch logs from {LOG_GROUP_NAME}...")
    print(f"Time range: {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")
    
    # Check if log group exists
    try:
        logs_client.describe_log_groups(logGroupNamePrefix=LOG_GROUP_NAME)
    except Exception as e:
        print(f"Error: Log group not found - {e}")
        return
    
    # Fetch logs with pagination
    all_events = []
    next_token = None
    
    while True:
        kwargs = {
            "logGroupName": LOG_GROUP_NAME,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 10000,
        }
        if next_token:
            kwargs["nextToken"] = next_token
        
        try:
            response = logs_client.filter_log_events(**kwargs)
            all_events.extend(response.get("events", []))
            next_token = response.get("nextToken")
            
            if not next_token:
                break
            print(f"  Fetched {len(all_events)} events so far...")
        except Exception as e:
            print(f"Error fetching logs: {e}")
            break
    
    # Save to file
    cw_output_dir = Path(output_dir) / "cloudwatch"
    cw_output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = cw_output_dir / f"step_function_logs_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(all_events, f, indent=2, default=str)
    
    print(f"Saved {len(all_events)} log events to {output_file}")
    
    # Also save a summary
    summary_file = cw_output_dir / f"step_function_summary_{timestamp}.txt"
    with open(summary_file, "w") as f:
        for event in all_events:
            ts = datetime.fromtimestamp(event["timestamp"] / 1000).isoformat()
            msg = event.get("message", "")[:200]
            f.write(f"[{ts}] {msg}\n")
    
    print(f"Saved summary to {summary_file}")


def download_s3_reports(output_dir: str = OUTPUT_DIR):
    """Download pipeline reports from S3 into a timestamped folder."""
    s3 = boto3.client("s3", region_name=REGION)

    # Create timestamped subfolder so each download is separate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(output_dir) / "s3" / timestamp

    print(f"Downloading S3 reports from {LOGS_BUCKET}...")
    print(f"Saving to: {base_dir}/")

    # All pipeline report prefixes
    REPORT_PREFIXES = [
        "fixed", "fixed_strong", "agentic", "rule_based", "rule_based_ss",
        "llm_no_memory", "llm_with_memory",
        "llm_with_memory_no_ss", "llm_no_memory_no_ss",
    ]

    total_count = 0
    paginator = s3.get_paginator("list_objects_v2")

    for prefix in REPORT_PREFIXES:
        out_dir = base_dir / prefix
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for page in paginator.paginate(Bucket=LOGS_BUCKET, Prefix=f"reports/{prefix}/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                filename = key.split("/")[-1]
                if filename and filename.endswith(".json"):
                    s3.download_file(LOGS_BUCKET, key, str(out_dir / filename))
                    count += 1
        if count > 0:
            print(f"  Downloaded {count} {prefix} reports")
        total_count += count

    # Download legacy reports (not in any known prefix)
    legacy_dir = base_dir / "legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_count = 0
    known = set(f"/{p}/" for p in REPORT_PREFIXES)
    for page in paginator.paginate(Bucket=LOGS_BUCKET, Prefix="reports/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if any(k in key for k in known):
                continue
            filename = key.split("/")[-1]
            if filename and filename.endswith(".json"):
                s3.download_file(LOGS_BUCKET, key, str(legacy_dir / filename))
                legacy_count += 1
    if legacy_count > 0:
        print(f"  Downloaded {legacy_count} legacy reports")
    total_count += legacy_count

    print(f"\nTotal: {total_count} reports downloaded to {base_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Download Step Function and pipeline logs")
    parser.add_argument("--source", choices=["cloudwatch", "s3", "all"], default="all",
                        help="Log source to download")
    parser.add_argument("--hours", type=int, default=24,
                        help="Hours of CloudWatch logs to download (default: 24)")
    parser.add_argument("--since", type=str,
                        help="Download logs since date (today, yesterday, or YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help="Output directory (default: logs)")
    args = parser.parse_args()
    
    output_dir = args.output
    
    since = parse_since(args.since) if args.since else None
    
    if args.source in ["cloudwatch", "all"]:
        download_cloudwatch_logs(hours=args.hours, since=since, output_dir=output_dir)
    
    if args.source in ["s3", "all"]:
        download_s3_reports(output_dir=output_dir)
    
    print(f"\nDone! Logs saved to {output_dir}/")


if __name__ == "__main__":
    main()
