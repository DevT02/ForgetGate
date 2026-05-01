# Next Benchmark Spec

This document defines the next project after the current controlled
multi-regime benchmark.

## Why a Phase 2 Benchmark Exists

The current ForgetGate benchmark is useful because it is:

- controlled
- reproducible
- good at exposing failure modes
- good for attack/defense iteration

But it is also intentionally narrow:

- mostly CIFAR-10
- mostly ViT-Tiny
- mostly LoRA-based unlearning implementations
- mostly class-forgetting

That is enough for a strong robustness-benchmark paper, but not enough for a
paper that claims broad coverage of modern methods or strong industry realism.

## Goal

Build a broader **machine-unlearning robustness benchmark** that keeps the
multi-regime attack philosophy while improving:

- method fidelity
- benchmark realism
- architecture coverage
- dataset breadth

## Non-Goals

This phase should not become:

- a vague “bigger is better” expansion
- another repo full of proxy methods
- a benchmark with weak provenance and unclear method fidelity

## Core Design Principles

1. Fidelity first
- only include methods whose paper/source can be pinned down clearly
- mark every method as:
  - `faithful`
  - `adapted`
  - `supporting`

2. Attack hierarchy stays fixed
- full-image conditional recovery
- one-patch local delivery
- frame delivery
- multi-patch delivery
- optional short-budget learned attack

3. Defense wins must be harder to claim
- no defense is positive unless it helps under more than one relevant regime or
  clearly improves the primary regime without introducing retain-side failure

4. Controlled and realistic layers should be separated
- do not mix toy and realistic settings into one table without explicit labels

## Phase 2 Benchmark Axes

### 1. Datasets

Minimum target:

- keep CIFAR-10 as the controlled microscope
- add at least one more realistic benchmark

Candidate second benchmark:

- CIFAR-100 or Tiny-ImageNet for vision breadth

Better longer-term benchmark:

- a dataset with more realistic deletion units:
  - user-level
  - sample-level
  - concept-level
  - document-level

### 2. Model Families

Minimum target:

- ViT-Tiny
- ResNet-18

Preferred expansion:

- one stronger ViT backbone
- one non-transformer baseline
- at least one fully finetuned setting per major method family

### 3. Forgetting Units

Current benchmark:

- single-class forgetting

Phase 2 should add:

- multi-class forgetting
- sample-level forgetting
- grouped / user-style forgetting if data allows

### 4. Methods

Only include methods if we can verify:

- actual paper/source
- actual algorithm identity
- actual valid forgetting checkpoint

Priority method types:

- safe canonical anchors
- recent robust-unlearning methods
- one or two repo-defined internal benchmark methods if labeled honestly

Methods should not enter main tables if:

- they are only proxies
- their checkpoints do not actually forget
- their literature provenance is unclear

## Concrete Method Selection Rule

For each candidate method, require all of:

1. source found
2. implementation identity understood
3. valid forgetting checkpoint trained
4. at least one seed with usable clean point

Then assign:

- `mainline`
- `supporting`
- `drop`

## Reporting Standard

Every benchmark row should report:

- clean forget accuracy
- clean retain accuracy
- forget recovery median
- retain-control median
- clean-success
- one-patch success
- frame success
- multi-patch success
- retain-to-forget failure where applicable
- number of seeds
- fidelity label

## What Counts as Success for Phase 2

Phase 2 is successful if it produces:

1. a cleaner method set than the current repo
2. at least one additional benchmark layer beyond CIFAR-10 single-class
3. at least one stronger architecture setting
4. a benchmark protocol that can be reused without re-explaining everything

## What Not to Repeat

Do not repeat these mistakes:

- convenience artifacts treated like canonical baselines
- repo-only names treated like known papers
- proxy rows mixed into main tables
- method count optimized over method fidelity

## Relationship to the Current Repo

This spec belongs in the current repo because:

- it grows directly out of the current benchmark
- the current repo already contains the protocol and attack stack
- splitting too early would fragment provenance

If Phase 2 accumulates substantial new code, datasets, and training artifacts,
then a later repo split or subproject may make sense.

For now, this should remain a roadmap attached to the current benchmark.
