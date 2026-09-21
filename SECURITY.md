# Security policy

Please report suspected security issues through
[GitHub private vulnerability reporting](https://github.com/HowardHsuuu/EEGFMLens/security/advisories/new).
Do not open a public issue with exploit details or sensitive recordings.

The latest tagged release is the supported version. EEGFMLens loads local model
constructors and checkpoints supplied by the caller; only use upstream source and
weights you trust. The built-in checkpoint helpers use weights-only loading and
strict state-dict validation.
