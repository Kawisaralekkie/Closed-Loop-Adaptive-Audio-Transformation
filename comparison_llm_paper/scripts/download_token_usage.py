#!/usr/bin/env python3
"""Download Bedrock token usage from CloudWatch and save to logs/.

Usage:
    python3 scripts/download_token_usage.py
    python3 scripts/download_token_usage.py --since 2026-04-19
    python3 scripts/download_token_usage.py --since today
"""

import argparse
import json
import os
from datetime import datetime, timezone, timedelta

import boto3

REGION = "ap-southeast-1"
MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
OUTPUT_DIR = "logs"


def fetch_metrics(start: datetime, end: datetime, period: int = 3600):
    cw = boto3.client("cloudwatch", region_name=REGION)
    metrics = ["InputTokenCount", "OutputTokenCount", "Invocations", "InvocationLatency"]
    stats = ["Sum", "Average", "SampleCount", "Minimum", "Maximum"]

    result = {"model_id": MODEL_ID, "region": REGION, "period_seconds": period,
              "start": start.isoformat(), "end": end.isoformat(), "fetched_at": datetime.now(timezone.utc).isoformat()}

    for metric in metrics:
        # With model dimension
        resp = cw.get_metric_statistics(
            Namespace="AWS/Bedrock", MetricName=metric,
            Dimensions=[{"Name": "ModelId", "Value": MODEL_ID}],
            StartTime=start, EndTime=end, Period=period, Statistics=stats)
        dps = sorted(resp["Datapoints"], key=lambda x: x["Timestamp"])
        result[metric] = [
            {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in dp.items()}
            for dp in dps
        ]

    # Summary
    inp_total = sum(dp.get("Sum", 0) for dp in result.get("InputTokenCount", []))
    out_total = sum(dp.get("Sum", 0) for dp in result.get("OutputTokenCount", []))
    inv_total = sum(dp.get("Sum", 0) for dp in result.get("Invocations", []))
    inv_count = sum(dp.get("SampleCount", 0) for dp in result.get("Invocations", []))

    result["summary"] = {
        "total_input_tokens": int(inp_total),
        "total_output_tokens": int(out_total),
        "total_tokens": int(inp_total + out_total),
        "total_invocations": int(inv_total),
        "avg_input_per_call": round(inp_total / max(inv_total, 1), 1),
        "avg_output_per_call": round(out_total / max(inv_total, 1), 1),
        "avg_total_per_call": round((inp_total + out_total) / max(inv_total, 1), 1),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Download Bedrock token usage from CloudWatch")
    parser.add_argument("--since", default="today", help="Start date: today, yesterday, or YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=1, help="Number of days to fetch (default: 1)")
    args = parser.parse_args()

    if args.since == "today":
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    elif args.since == "yesterday":
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    else:
        start = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    end = start + timedelta(days=args.days)

    print(f"Fetching Bedrock token usage: {start.date()} to {end.date()}")
    data = fetch_metrics(start, end)

    s = data["summary"]
    print(f"\nSummary:")
    print(f"  Invocations: {s['total_invocations']:,}")
    print(f"  Input tokens: {s['total_input_tokens']:,} (avg {s['avg_input_per_call']:,.1f}/call)")
    print(f"  Output tokens: {s['total_output_tokens']:,} (avg {s['avg_output_per_call']:,.1f}/call)")
    print(f"  Total tokens: {s['total_tokens']:,} (avg {s['avg_total_per_call']:,.1f}/call)")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bedrock_token_usage_{start.strftime('%Y%m%d')}_{ts}.json"
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
