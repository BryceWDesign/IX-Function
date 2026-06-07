# IX-Function

IX-Function is a source-available evaluation repo for testing whether a learned causal structure can transfer across different domains under evidence, uncertainty, falsification, and human-review gates.

It is designed as a serious donor component for an IX-CognitionKernel Wave 6 attempt, not as an AGI system by itself.

The central question IX-Function is built around:

> Can the system learn a causal structure in one domain, carry it into a different domain, make a useful pre-outcome prediction, receive reality feedback, revise its model, preserve uncertainty, and improve future decisions without bespoke hand-scripting?

IX-Function attempts to make that question testable, reviewable, falsifiable, and replayable.

## Claim boundary

IX-Function does **not** claim:

- AGI
- certified AGI
- independent AGI validation
- deployment authority
- production readiness
- safety certification
- self-approval
- unsupervised operational autonomy

IX-Function may support a much narrower claim:

> Bounded cross-domain causal-transfer evidence was produced under pre-outcome prediction, reality-delta scoring, uncertainty preservation, falsification, negative controls, model-output review, and human-review boundaries.

That is the maximum claim boundary unless independent reviewers reproduce stronger evidence.

## What IX-Function does

IX-Function creates a full evidence chain for causal transfer:

1. Define a source domain and target domain.
2. Define a reusable causal function.
3. Record source-domain learning evidence.
4. Map causal slots into a different target domain.
5. Commit to a prediction before the target outcome is observed.
6. Score the observed target outcome against the committed prediction.
7. Convert reality feedback into future behavior rules.
8. Preserve uncertainty instead of laundering ambiguity into certainty.
9. Apply falsification criteria and kill gates.
10. Run negative controls to prevent theater.
11. Serialize deterministic evidence packets.
12. Build donor handoffs for the larger IX stack.
13. Review model-provider interpretations as untrusted input.
14. Build an independent replication packet.
15. Gate bounded Wave 6 review language.

## Repository status

This repo is structured as an evaluation harness. It should be treated as a research artifact and evidence tool, not as an operational autonomy system.

Expected local checks:

```
python -m ruff check .
python -m mypy src tests
python -m pytest
```

No result should be publicly described as passing unless these checks actually pass in the current working tree.

## License

IX-Function is distributed under a Source-Available Evaluation License.

The intended posture is:

- public technical evaluation is allowed under the license terms
- commercial use requires prior written permission
- production use requires prior written permission
- hosted service use requires prior written permission
- government operational use requires prior written permission
- procurement, contractor, funded-pilot, or derivative commercial use requires a separate written license

See `LICENSE` for the binding license text.

## Core architecture

### Domain layer

The domain layer describes bounded source and target domains.

Relevant module:

```
src/ix_function/domain.py
```

It models:

- domain profiles
- observable roles
- observable value kinds
- source-target domain pairs
- cross-domain checks
- validation gates

A transfer trial is not meaningful unless the source and target are genuinely different enough to pressure transfer.

### Observation layer

The observation layer records measurable state before and after intervention.

Relevant module:

```
src/ix_function/observation.py
```

It models:

- measured values
- baseline snapshots
- intervention records
- outcome records
- validation against declared domain observables

The outcome must reference the intervention it follows. This prevents loose narrative scoring.

### Causal-function layer

The causal-function layer defines the portable structure IX-Function is trying to transfer.

Relevant module:

```
src/ix_function/causal_function.py
```

It models:

- causal families
- causal variable slots
- causal mechanisms
- reusable causal functions
- causal signatures

A causal function is intentionally abstract enough to transfer, but constrained enough to be falsifiable.

### Source-learning layer

The source-learning layer records whether the causal function has bounded support in the source domain.

Relevant module:

```
src/ix_function/source_learning.py
```

It models:

- source learning trials
- source evidence records
- support status
- confidence deltas
- uncertainty notes
- blocking errors

Source evidence does not prove universality. It only supports attempting a bounded transfer trial.

### Mapping layer

The mapping layer proposes how abstract causal slots bind to concrete target-domain observables.

Relevant module:

```
src/ix_function/mapping.py
```

It models:

- slot candidates
- slot mappings
- transfer mappings
- mapping quality
- mapping warnings
- ambiguity scores

A mapping can be:

- complete
- ambiguous
- insufficient

Insufficient mappings cannot support prediction.

### Prediction layer

The prediction layer records committed target-domain predictions before the outcome is observed.

Relevant module:

```
src/ix_function/prediction.py
```

It models:

- predicted observables
- prediction direction
- numeric ranges
- tolerances
- prediction readiness
- outcome-leakage blockers

A valid prediction must be created before outcome observation and must not reference forbidden outcome identifiers.

### Reality-delta layer

The reality-delta layer compares prediction against observed outcome.

Relevant module:

```
src/ix_function/reality_delta.py
```

It models:

- observable deltas
- direction matches
- range matches
- outcome status
- confidence deltas
- uncertainty notes
- unscorable outcomes

Reality-delta scoring is where attractive transfer narratives are separated from measured support or failure.

### Learning-update layer

The learning layer converts scored reality feedback into future behavior changes.

Relevant module:

```
src/ix_function/learning.py
```

It models:

- revised confidence
- confidence bands
- learning dispositions
- future planning rules
- quarantine decisions

IX-Function is not useful if outcomes only create receipts. Reality feedback must change future behavior.

### Uncertainty layer

The uncertainty layer preserves ambiguity, assumptions, mapping warnings, measurement issues, failed outcomes, and blocking errors.

Relevant module:

```
src/ix_function/uncertainty.py
```

It models:

- uncertainty items
- uncertainty ledgers
- severity
- lifecycle state
- uncertainty gates
- claim-strength gates

The system must preserve uncertainty instead of pretending certainty.

### Falsification layer

The falsification layer applies explicit kill criteria.

Relevant module:

```
src/ix_function/falsification.py
```

Default criteria include:

- cross-domain transfer requirement
- usable mapping requirement
- outcome support requirement
- blocking uncertainty requirement
- future behavior update requirement
- reproducibility reference requirement

A killed claim is a useful result. It means the harness is allowed to block itself.

### Negative-control layer

The negative-control layer checks that bad evidence is not promoted.

Relevant module:

```
src/ix_function/negative_control.py
```

Default negative controls include:

- insufficient mapping
- same-domain theater
- expected failure
- outcome leakage
- shuffled mapping

Negative controls are scenario-specific. A clean positive trial is not penalized for not being a negative-control fixture.

### Trial orchestration layer

The trial layer binds the full evidence chain.

Relevant module:

```
src/ix_function/trial.py
```

The orchestration path is:

```
source learning
-> target mapping
-> pre-outcome prediction
-> reality delta
-> learning update
-> uncertainty ledger
-> falsification ledger
-> negative-control suite
-> final trial status
```

Trial statuses include:

- bounded evidence allowed
- blocked
- downgraded
- invalid
- retest required

## Evidence packets

Relevant module:

```
src/ix_function/evidence.py
```

IX-Function serializes deterministic evidence artifacts with SHA-256 digests.

The evidence packet includes:

- trial summary
- source evidence
- mapping
- prediction readiness
- reality delta
- learning update
- uncertainty ledger
- uncertainty gate
- falsification ledger
- negative-control suite
- anti-theater gate

Digest binding helps reviewers detect mutation, but a digest does not prove truth.

## Donor handoffs

IX-Function includes donor handoff modules for the larger IX stack.

### IX-CognitionKernel handoff

Relevant module:

```
src/ix_function/kernel_handoff.py
```

Creates:

- Kernel belief update candidates
- memory update candidates
- skill candidates
- uncertainty-bound reuse conditions

The Kernel handoff does not promote memory as truth. It exports bounded evidence for review.

### IX-BlackFox-WorldTwin handoff

Relevant module:

```
src/ix_function/worldtwin_handoff.py
```

Creates:

- scenario packets
- variable bindings
- prediction bindings
- outcome-delta bindings
- adaptation guidance

The WorldTwin handoff supports replay and scenario retesting.

### IX-IntentRealityLoop handoff

Relevant module:

```
src/ix_function/intent_loop_handoff.py
```

Creates:

- intent bindings
- permission bindings
- reality-feedback bindings
- memory-binding requests

The handoff preserves the rule that decoded or inferred intent is not permission.

### IX-BlackFox handoff

Relevant module:

```
src/ix_function/blackfox_handoff.py
```

Creates:

- governed review bundles
- evidence bindings
- policy gates
- human-approval requirements

BlackFox treats transfer output as untrusted input.

### IX-Autonomy-Assurance-Case-Runtime handoff

Relevant module:

```
src/ix_function/assurance_handoff.py
```

Creates:

- bounded assurance claims
- safety gates
- trace links
- provenance bindings
- human-authority requirements

The assurance handoff does not certify safety, autonomy, deployment, or AGI.

### IX declarative contract handoff

Relevant module:

```
src/ix_function/ix_contract_handoff.py
```

Creates:

- review contracts
- preconditions
- declarative steps
- claim boundaries
- human authority requirements

The contract is not an execution grant.

### Integrated donor handoff bundle

Relevant module:

```
src/ix_function/handoff_bundle.py
```

Combines all donor handoffs into one validation object:

- Kernel
- WorldTwin
- IntentRealityLoop
- BlackFox
- Assurance
- IX contract

The integrated bundle can be ready for bounded Wave 6 review, failure review, or blocked.

## Model-provider review

Relevant module:

```
src/ix_function/model_review.py
```

Model-provider review treats outputs from LLMs or other model providers as untrusted interpretations.

It checks:

- model output shape
- cited evidence references
- required evidence references
- uncertainty acknowledgement
- human-review acknowledgement
- overclaiming language
- blocked claim kinds
- multi-provider coverage

A model cannot override:

- evidence packets
- falsification gates
- uncertainty ledgers
- human authority
- claim boundaries

Blocked claims include:

- AGI proof
- deployment authority
- production readiness

## Independent replication and Wave 6 review gate

Relevant module:

```
src/ix_function/wave6_review.py
```

The final code-level gate builds:

- independent replication packet
- replication steps
- expected replay outputs
- kill criteria
- Wave 6 review gate result

The Wave 6 review gate can decide:

- ready for bounded Wave 6 review
- require independent replication
- blocked

Even when ready, the allowed claim is only bounded review-candidate evidence.

## Minimal usage shape

The tests include reusable fixtures in:

```
tests/fixtures.py
```

A typical end-to-end flow is:

```
from ix_function.evidence import build_trial_evidence_packet
from ix_function.handoff_bundle import build_integrated_handoff_bundle
from ix_function.model_review import build_model_provider_review_report
from ix_function.trial import run_transfer_trial
from ix_function.wave6_review import build_wave6_review_gate

# Build or load a TransferTrialInput.
trial_result = run_transfer_trial(trial_input)

evidence_packet = build_trial_evidence_packet(trial_result)

integrated_bundle = build_integrated_handoff_bundle(
    result=trial_result,
    evidence_packet=evidence_packet,
)

model_review_report = build_model_provider_review_report(
    report_id=f"{trial_result.trial_id}:model-review",
    result=trial_result,
    evidence_packet=evidence_packet,
    outputs=model_outputs,
)

wave6_gate = build_wave6_review_gate(
    result=trial_result,
    evidence_packet=evidence_packet,
    integrated_bundle=integrated_bundle,
    model_review_report=model_review_report,
)
```

The output of `wave6_gate.permits_bounded_wave6_review()` should never be interpreted as AGI proof.

## Development

Recommended local setup:

```
python -m venv .venv
. .venv/Scripts/activate
python -m pip install -e ".[dev]"
```

Recommended validation:

```
python -m ruff check .
python -m mypy src tests
python -m pytest
```

Use the repository only after the current working tree passes its checks.

## What success would mean

A successful IX-Function trial would mean:

- a source causal function had bounded support
- the function mapped into a genuinely different target domain
- a prediction was committed before the outcome
- reality feedback supported the prediction
- confidence changed conservatively
- future behavior rules were updated
- uncertainty remained visible
- falsification criteria did not kill the claim
- negative controls did not expose theater
- deterministic evidence was serialized
- donor handoffs validated
- model-provider outputs did not overclaim
- independent replay packet was ready

That still would not prove AGI.

It would be evidence that the missing IX-Function piece is working as a cross-domain causal-transfer harness.

## What failure would mean

A failed IX-Function trial is not useless.

Failure can show:

- mapping did not generalize
- target outcome contradicted the prediction
- uncertainty blocked stronger claims
- falsification criteria killed the claim
- model output overclaimed
- negative controls exposed theater
- evidence was not replayable
- independent replication was not ready

A failure is valuable if it is preserved honestly and prevents overclaiming.

## Public positioning

Careful public wording:

> IX-Function is a source-available evaluation harness for bounded cross-domain causal-transfer evidence. It tests whether a learned causal structure can transfer into a different domain through pre-outcome prediction, reality feedback, uncertainty preservation, falsification, negative controls, and human-review gates.

Avoid saying:

- “IX-Function is AGI.”
- “IX-Function proves AGI.”
- “IX-Function is independently validated.”
- “IX-Function is production ready.”
- “IX-Function authorizes deployment.”
- “IX-Function removes human review.”

## Current intended role in Wave 6

IX-Function is intended to serve as the missing causal-transfer and reality-feedback bridge for an IX-CognitionKernel Wave 6 attempt.

Its strongest role is to test whether a causal structure can:

- transfer across genuinely different domains
- make useful pre-outcome predictions
- receive measurable feedback
- revise confidence and future behavior
- preserve uncertainty
- fail safely
- resist model hallucination and overclaiming
- support independent replay

That is the honest contribution.

It is not the whole AGI claim.

## Copyright

Copyright © 2026 Bryce Lovell.

All rights reserved except as expressly granted in the license.
