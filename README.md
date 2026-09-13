# ITCS355 Lab 1 — Reproducible Training

> **Course materials live in [`course/`](course/README.md)** — syllabus, slides, the faculty
> specification, all five lab handouts, and the project brief. Every document is Markdown and
> renders on GitHub, diagrams included. New to the repo? Start with the
> [portability reference](course/reference/cloud-portability-reference.md).
> Keep this block when you edit the rest of this file; it is not part of the Lab 1 deliverable.

Predicting machine failure within 7 days from sensor readings. The model is not the point;
whether a stranger can reproduce it is.

> **This README is graded.** A grader with Docker and nothing else from your setup runs one
> command and compares the result against the claim below. Keep the claim and execution
> instructions synchronized before submitting.

---

## Reproduce

```bash
make reproduce
```

expected test_roc_auc: 0.848 ± 0.010

Runtime: about 40 seconds on 4 cores. No cloud account or credentials needed for this command —
that is deliberate, and it is why a grader can run it.

The claim is intentionally kept to three decimals with a tolerance of 0.010. `make verify` parses
this line and compares it with the metric produced by the pinned-seed container run. The tolerance
covers the observed cross-platform floating-point variation, not an uncontrolled change of seed.

---

## The problem

240 machines, 25 readings each, 6 sensor features, binary target `failed_within_7d` with a
positive rate near 12%.

Machines have persistent characteristics — a hot-running machine reads hot in every row. So the
train/validation/test split is **grouped by `machine_id`**: every reading from one machine lands
in exactly one partition. Splitting row-wise instead lets the model memorise the machine and
reports a validation score that will never survive production. `tests/test_data.py` asserts this
property holds, and Lab 4 turns it into a CI gate.

Bringing your own dataset is allowed. To do that, update `scripts/make_dataset.py`, the schema in
`src/data.py`, and keep every test passing.

---

## Layout

```
src/          Layer 1 — provider-neutral. No SDKs, no bucket names, no absolute paths.
cloudlayer/   Layer 3 — the only place a provider SDK may be imported.
scripts/      Dataset generation, cloud check, portability audit, metric verification.
tests/        Data contract tests and split property tests.
```

`src/config.py` is the single point of environment knowledge. Everything else reads from it.
`make portability-audit` enforces the rule; it fails the build if a provider string appears in
`src/` or `tests/`.

---

## Setup

```bash
cp cloud.env.example cloud.env      # fill in, never commit
make setup
make cloud-check                    # eight slots, all PASS
make data                           # generate the dataset
make test                           # 10 tests, all passing
```

Post your `make cloud-check` output in the course channel before Session 1.

---

## Lab 1 implementation

The Lab 1 decisions are implemented in this checkout:

| Where | What |
|---|---|
| `requirements.txt` | Hash-pinned output generated from `requirements.in` |
| `Dockerfile` | Multi-stage, non-root, digest-pinned base and hash-enforced install |
| `cloudlayer/aws.py` | S3 upload/download and ECR login, push, tagging, and digest return |
| `src/train.py` | Seed, Git SHA, raw fingerprint, DVC hash, metrics, and model artifact logging |

Then:

```bash
make image-push        # image reaches your registry, digest-pinned
dvc init && dvc remote add -d storage ${BLOB_URI}/dvc
dvc add data/raw && dvc push
```

Run five or more tracked runs varying something meaningful — not five identical runs with
different seeds.

---

## Reproducibility trade-off

Three things pin your build: hashed dependencies, a digest-pinned base image, and controlled
seeds. Under real time pressure you would keep some and drop others.

I would drop dependency hashes first, while keeping the digest pin and seed control. The version
pins would still constrain the dependency graph and the digest would keep the OS and interpreter
fixed, so the build should continue to work. What breaks is supply-chain reproducibility: a package
index could serve a different wheel for the same version, changing code or introducing a vulnerability.
The next improvement would be restoring hashes before changing model or data behavior, because that
failure is silent and difficult to diagnose.

---

## Notes for the grader

This checkout uses the AWS adapter. `make reproduce` is intentionally cloud-free: it runs from the
checked-in raw data and the pinned Docker image. Cloud completion still requires filling `cloud.env`,
granting the AWS identity access to the configured S3 prefix and ECR repository, then running
`make image-push`, `dvc push`, and the five-run MLflow study. No credentials are committed or copied
into an image layer.

---

## Checklist before you submit

- [ ] `make reproduce` works from a fresh clone, on a machine that is not yours
- [x] `make verify` passes against the final claim line
- [x] `make test` — all tests pass
- [x] `make portability-audit` — clean
- [ ] Image builds for `linux/amd64` and is pushed, digest-pinned
- [ ] `dvc push` completed; a grader can `dvc pull`
- [x] Five or more tracked runs with params, metrics, data fingerprint, DVC hash, and commit SHA
- [x] No instruction placeholders remain in the Lab 1 README
- [x] `git log -p | grep -i -E "secret|password|AKIA|BEGIN PRIVATE"` returns nothing

That last check is not optional. A credential in Git history is an automatic deduction in this
course, and rotating it is your responsibility, not the grader's.
