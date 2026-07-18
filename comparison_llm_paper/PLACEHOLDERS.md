# Credential Placeholders

This is the publish-ready copy of the project. All AWS-specific credentials and
account-scoped identifiers have been replaced with placeholders. Fill these in
(via environment variables, CDK context, or direct edits) before deploying.

| Placeholder | Meaning | Example format |
|---|---|---|
| `<AWS_ACCOUNT_ID>` | AWS account ID used in ARNs | 12-digit number |
| `<PRIMARY_ACCOUNT_ID>` | Primary account (source of S3 replication) | 12-digit number |
| `<BACKUP_ACCOUNT_ID>` | Backup account (replication destination) | 12-digit number |
| `<KMS_KEY_ID>` | KMS key UUID used for bucket encryption | UUID |
| `<RAW_AUDIO_BUCKET>` | S3 bucket for raw audio uploads | bucket name |
| `<PROCESSED_AUDIO_BUCKET>` | S3 bucket for processed/blurred audio | bucket name |
| `<LOGS_BUCKET>` | S3 bucket for logs / decision logs | bucket name |
| `<KNOWLEDGE_BASE_BUCKET>` | S3 bucket for Bedrock Knowledge Base artifacts | bucket name |

## Notes

- Several scripts already read identifiers from environment variables
  (`AWS_ACCOUNT_ID`, `AWS_REGION`, `RAW_AUDIO_BUCKET`). Set those before running.
- `infra/cdk.context.json` (cached account/AZ lookups) was intentionally omitted;
  CDK regenerates it on the next `cdk synth`.
- Build artifacts and environments were excluded from this copy:
  `.venv/`, `infra/cdk.out/`, `infra/cdk.out.tmp/`, `.pytest_cache/`,
  `.hypothesis/`, `__pycache__/`, and output/data folders (`logs/`, `plots/`,
  `model_out/`, `multi_threshold_output/`, `parameter/`, `data/`).
