# Team Workflow

## Protected integration branch

`main` represents reviewed/stable integration state.

No developer should intentionally implement features directly on `main`.

## Rohan branch

`feat/rohan-ai-core`

## Person 2 branch

`feat/person2-ui-demo`

## Feature workflow

Every implementation task follows:

```text
main
 ↓
feature branch
 ↓
implementation
 ↓
tests
 ↓
commit
 ↓
push
 ↓
PR to main
 ↓
Brain review
 ↓
approval
 ↓
merge
```

## Review gate

PRs must NOT be merged merely because tests pass.

The project Brain/reviewer must inspect:

* scope
* implementation
* tests
* regressions
* evidence
* architecture impact

before approving integration.

## Branch hygiene

Do not:

* force-push main
* rewrite shared history
* commit secrets
* commit model checkpoints
* commit datasets
* commit `.env`
* commit API keys
* commit Twilio credentials
