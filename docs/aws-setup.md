# Deploying to AWS

This sets up the cloud side of the pipeline:

```
GitHub push ─► CI (tests + image build) ─► Deploy workflow
                                              │
          S3 (trained model) ─────────────────┤
                                              ▼
                     ECR (Docker image) ─► ECS Fargate (API) ─► RDS PostgreSQL
                                              │
                                        CloudWatch Logs
```

Until you finish these steps the Deploy workflow is **skipped**, not failed, so CI stays green.

> **Cost warning.** Fargate and RDS are billed by the hour. A small setup
> (1 vCPU / 3 GB task + `db.t4g.micro`) is roughly €1–2 per day. Run it for
> demos, then delete it (see the end of this page). Set a billing alarm in
> AWS Budgets before you start.

All commands use the AWS CLI with region `eu-west-1` (Ireland). Replace
`ACCOUNT_ID` with your 12-digit account id (`aws sts get-caller-identity`).

```bash
export AWS_REGION=eu-west-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export GITHUB_REPO=manu5a/distilbert-ticket-classifier
```

## 1. S3 bucket for the trained model

The model (~260 MB) is too big for git, so it lives in S3.

```bash
aws s3 mb s3://ticket-classifier-models-$ACCOUNT_ID --region $AWS_REGION
```

## 2. ECR repository for the Docker image

```bash
aws ecr create-repository --repository-name ticket-classifier --region $AWS_REGION
```

## 3. PostgreSQL on RDS

Create it in the console (RDS → Create database) – it's easier the first time:

- Engine: PostgreSQL 16, template **Free tier** / **Dev/Test**
- Instance: `db.t4g.micro`, storage 20 GB
- DB name `tickets`, username `tickets`, a strong password
- Public access: **No**
- Note the security group it creates (call it `rds-sg`)

Store the connection string as a secret (ECS injects it as `DATABASE_URL`):

```bash
aws secretsmanager create-secret --name ticket-classifier/database-url \
  --secret-string "postgresql+psycopg://tickets:YOUR_PASSWORD@YOUR_RDS_ENDPOINT:5432/tickets"
```

## 4. IAM roles

**a) Task execution role** – lets ECS pull the image, write logs and read the secret.

```bash
aws iam create-role --role-name ticket-classifier-execution-role \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name ticket-classifier-execution-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
aws iam put-role-policy --role-name ticket-classifier-execution-role --policy-name read-db-secret \
  --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[
    {\"Effect\":\"Allow\",\"Action\":\"secretsmanager:GetSecretValue\",\"Resource\":\"arn:aws:secretsmanager:$AWS_REGION:$ACCOUNT_ID:secret:ticket-classifier/*\"},
    {\"Effect\":\"Allow\",\"Action\":\"logs:CreateLogGroup\",\"Resource\":\"*\"}]}"
```

**b) GitHub deploy role (OIDC)** – GitHub Actions logs in to AWS with a
short-lived token, so no AWS keys are stored in GitHub.

```bash
aws iam create-open-id-connect-provider --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com

aws iam create-role --role-name ticket-classifier-github-deploy \
  --assume-role-policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{
    \"Effect\":\"Allow\",
    \"Principal\":{\"Federated\":\"arn:aws:iam::$ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com\"},
    \"Action\":\"sts:AssumeRoleWithWebIdentity\",
    \"Condition\":{
      \"StringEquals\":{\"token.actions.githubusercontent.com:aud\":\"sts.amazonaws.com\"},
      \"StringLike\":{\"token.actions.githubusercontent.com:sub\":\"repo:$GITHUB_REPO:*\"}}}]}"

aws iam put-role-policy --role-name ticket-classifier-github-deploy --policy-name deploy \
  --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[
    {\"Effect\":\"Allow\",\"Action\":\"ecr:GetAuthorizationToken\",\"Resource\":\"*\"},
    {\"Effect\":\"Allow\",\"Action\":[\"ecr:BatchCheckLayerAvailability\",\"ecr:InitiateLayerUpload\",\"ecr:UploadLayerPart\",\"ecr:CompleteLayerUpload\",\"ecr:PutImage\",\"ecr:BatchGetImage\"],
     \"Resource\":\"arn:aws:ecr:$AWS_REGION:$ACCOUNT_ID:repository/ticket-classifier\"},
    {\"Effect\":\"Allow\",\"Action\":[\"s3:ListBucket\",\"s3:GetObject\",\"s3:PutObject\",\"s3:DeleteObject\"],
     \"Resource\":[\"arn:aws:s3:::ticket-classifier-models-$ACCOUNT_ID\",\"arn:aws:s3:::ticket-classifier-models-$ACCOUNT_ID/*\"]},
    {\"Effect\":\"Allow\",\"Action\":[\"ecs:RegisterTaskDefinition\",\"ecs:DescribeTaskDefinition\",\"ecs:UpdateService\",\"ecs:DescribeServices\"],\"Resource\":\"*\"},
    {\"Effect\":\"Allow\",\"Action\":\"iam:PassRole\",\"Resource\":\"arn:aws:iam::$ACCOUNT_ID:role/ticket-classifier-execution-role\"}]}"
```

## 5. Edit the task definition

In `deploy/aws/task-definition.json` replace every `ACCOUNT_ID` and `REGION`
with your values, then commit. To register a first version by hand
(the service in step 6 needs one):

```bash
aws ecs register-task-definition --cli-input-json file://deploy/aws/task-definition.json
```

The `image` field is filled in automatically by the Deploy workflow; for this
first manual registration ECS doesn't check it.

## 6. ECS cluster and service

```bash
aws ecs create-cluster --cluster-name ticket-classifier
```

Then in the console: **ECS → Clusters → ticket-classifier → Create service**

- Launch type **Fargate**, task definition `ticket-classifier`, desired tasks **1**
- Networking: default VPC, public subnets, **Public IP: on**
- New security group `api-sg` allowing inbound TCP **8000** from your IP
- Service name `ticket-classifier-api`

Finally allow the API to reach the database: edit `rds-sg` and add an inbound
rule for TCP **5432** from `api-sg`.

> For a production setup you'd put an Application Load Balancer with HTTPS in
> front and use private subnets. Not needed for a portfolio demo.

## 7. Connect GitHub

GitHub repo → **Settings → Secrets and variables → Actions**

| Kind | Name | Value |
|---|---|---|
| Secret | `AWS_ROLE_ARN` | `arn:aws:iam::ACCOUNT_ID:role/ticket-classifier-github-deploy` |
| Variable | `AWS_DEPLOY_ENABLED` | `true` |
| Variable | `AWS_REGION` | `eu-west-1` |
| Variable | `ECR_REPOSITORY` | `ticket-classifier` |
| Variable | `ECS_CLUSTER` | `ticket-classifier` |
| Variable | `ECS_SERVICE` | `ticket-classifier-api` |
| Variable | `MODEL_S3_URI` | `s3://ticket-classifier-models-ACCOUNT_ID/ticket-classifier` |

## 8. Train, then deploy

1. **Actions → Train model → Run workflow.** It fine-tunes DistilBERT, prints
   the test-set report and uploads the model to S3.
2. **Actions → Deploy to AWS → Run workflow** (after this, every green CI run
   on `main` deploys automatically).
3. Find the task's public IP (ECS → service → Tasks → task → Public IP) and open
   `http://PUBLIC_IP:8000/docs`.

Logs: CloudWatch → Log groups → `/ecs/ticket-classifier`.

## Tear everything down

```bash
aws ecs update-service --cluster ticket-classifier --service ticket-classifier-api --desired-count 0
aws ecs delete-service --cluster ticket-classifier --service ticket-classifier-api --force
aws ecs delete-cluster --cluster ticket-classifier
aws rds delete-db-instance --db-instance-identifier YOUR_DB_ID --skip-final-snapshot
aws ecr delete-repository --repository-name ticket-classifier --force
aws s3 rb s3://ticket-classifier-models-$ACCOUNT_ID --force
aws secretsmanager delete-secret --secret-id ticket-classifier/database-url --force-delete-without-recovery
```
