# Deployment: designed, never applied

> **This project was never deployed to AWS.** No stack was created, no image was pushed, no endpoint
> exists, and no cloud measurement appears anywhere in this repository. Everything below is a design
> written out in full, and should be read as such.

## Why it was not deployed

The OpenCV AI Competition's **$150 compute-grant round closed before we could enter it**, and we
chose not to put a personal credit card behind a student competition entry. There was no AWS account
to deploy into, so there was nothing to measure.

The alternative would have been to deploy briefly on a personal card and quote the numbers. We did
not, and we would rather hand in an honest gap than a number we had to caveat.

**What this means for the claims in this project:**

* There are **no cloud benchmarks**, no cold-start timings and no cost-per-frame figures.
* There are **no Graviton numbers**. `template.yaml` specifies `arm64` because it is the right
  default for this workload, not because it was measured against x86. No comparison was run.
* There is **no live endpoint**. The `ApiEndpoint` output in `template.yaml` is a CloudFormation
  substitution that would produce a URL if the stack existed. It does not exist. Do not quote it.
* The IaC is **unvalidated against a real account**. `sam validate` checks syntax; it does not check
  that the stack would come up, that the budget action would attach, or that the container would fit
  the image size limit.

Every measurement in [`report.md`](report.md) was produced locally, on a Windows 11 laptop, from
cached FIgLib imagery.

## What the design is

![Pipeline](architecture.png)

```
camera / site uploader
        |  PUT <camera>/<unix_ts>.jpg
        v
   S3 frame bucket  ----(ObjectCreated)----> Lambda (container, OpenCV 5, arm64)
        |                                          |
   14-day lifecycle                           reference cache
                                              + fitted model
                                                   |
   API Gateway (REST) --POST /score--------------> |
   2000 req/day quota, 5 rps                       v
                                            CloudWatch metrics
                                            SNS -> operator
```

Files:

| File | What it is |
| --- | --- |
| `infra/template.yaml` | SAM stack: buckets, function, API, alarms, budget action |
| `infra/Dockerfile` | arm64 Lambda container image, never built |
| `infra/requirements-lambda.txt` | Runtime subset of `pyproject.toml` |
| `infra/handler.py` | Lambda entry point, reusing `sentinel` unchanged, never invoked |

### The three decisions that actually matter

**1. A container, not a zip.** OpenCV 5 plus numpy and scikit-learn exceeds the 250 MB unzipped
layer limit. `opencv-python-headless` is also the correct build for Lambda: no GUI libraries, so no
`libGL.so.1` to install and nothing expecting an X display.

**2. The reference is state, and that is the hard part.** Scoring one frame needs tens of minutes of
that camera's earlier frames. A naive stateless function would re-read the whole window from S3 on
every request. `handler.py` builds the reference once per camera, caches it in the warm execution
context for an hour, and relies on the **anchored** strategy from
[`report.md`](report.md) section 6: an anchored reference does not need rebuilding as the hour
advances, which is exactly what makes this affordable. The bug fix and the cost model are the same
decision.

**3. Reserved concurrency is the most important line in the file.** `ReservedConcurrentExecutions:
5`. Without it a camera stuck in a retry loop fans out to the account concurrency limit and spends
the whole budget in minutes. On a student account the controls exist to bound the bill, not to scale
the system.

### Cost controls, because this is a student account

| Control | Setting | Why |
| --- | --- | --- |
| Reserved concurrency | 5 | Caps the worst case at five frames in flight |
| API usage plan quota | 2,000 req/day | A hard daily ceiling, not a capacity plan |
| API throttle | 5 rps, burst 10 | Stops a loop from consuming the daily quota in a minute |
| S3 lifecycle | expire frames at 14 days | Frames are only useful as recent history |
| Budget notification | 80% of $20 | Early warning |
| Budget action | 100% of $20, **manual approval** | Attaches a deny policy to the execution role |

The budget action is deliberately `ApprovalModel: MANUAL`. Automatically switching off a
fire-detection system because a bill crossed a line is its own kind of hazard; a human should decide,
having been told.

### Alarms

Three, and the second is the one people forget:

* **Errors** above 5 in five minutes: the function is failing.
* **No invocations for an hour**: the pipeline has gone silent. A silent detector is
  indistinguishable from a quiet fire season, so silence has to page somebody.
* **Throttles**: reserved concurrency is being hit, so frames are being dropped rather than scored
  late. Raise concurrency or reduce camera cadence.

## What it would cost to find out

If someone did deploy this, these are the numbers worth measuring, and none of them are known:

1. Cold-start time for a ~900 MB OpenCV container on arm64, which decides whether reserved
   concurrency or provisioned concurrency is the right control.
2. Wall-clock per frame in Lambda against the ~1.5 s measured locally.
3. Whether a cached reference survives long enough in warm containers to matter, or whether the
   reference belongs in EFS or a sidecar instead.
4. arm64 against x86_64 on this specific numpy and OpenCV mix, which is the only honest way to make
   a Graviton claim.

## If it were deployed

Not run, and recorded here so the design is complete rather than as an instruction anyone should
follow unverified:

```bash
aws ecr create-repository --repository-name smoke-sentinel
docker buildx build --platform linux/arm64 -f infra/Dockerfile -t smoke-sentinel .
# tag and push to ECR, then:
sam deploy --template infra/template.yaml \
  --stack-name smoke-sentinel \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides ImageUri=<ecr-uri> AlertEmail=<email>
# upload the fitted model and per-camera calibration
aws s3 cp evaluation/model.joblib s3://<artefact-bucket>/model/model.joblib
```

The per-camera calibration statistics that `handler.py` reads from
`calibration/<camera>.json` are produced by an offline job that does not exist yet. Without them the
handler returns `"calibrated": false` and the probability is not comparable to the reported operating
point, which is stated in the response rather than hidden.
