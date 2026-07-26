# Security policy

## Supported versions

Only the latest release is supported. Fixes are published as new releases rather
than backported.

## Reporting a vulnerability

Please report security issues privately using GitHub's
[private vulnerability reporting](https://github.com/tharindu326/physiotrack/security/advisories/new)
rather than opening a public issue. Include a description, affected version
(`python -c "import physiotrack; print(physiotrack.__version__)"`), and reproduction
steps. Expect an initial response within 14 days.

## Scope notes

Two properties of this project are worth understanding before reporting:

- **Model weights are downloaded at runtime** from Hugging Face and Ultralytics, and
  are loaded as PyTorch checkpoints. Downloads are streamed to a temporary file and
  renamed atomically, and the transfer size is checked, but checkpoint contents are
  **not** currently verified against a published hash. Only point the library at
  weight sources you trust.
- **PhysioTrack processes video of people and derives physiological signals.** It is
  a research tool, not a medical device, and it performs no anonymisation. Handling
  of recorded subjects' data, consent, and retention is the responsibility of the
  deploying party.
