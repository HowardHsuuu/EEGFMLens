"""Errors that identify invalid or unsupported experiments."""


class EEGLensError(RuntimeError):
    """Base exception for execution-contract violations."""


class ValidationError(EEGLensError):
    """Input, metadata or replacement is incompatible."""


class UnsupportedSiteError(EEGLensError):
    """An adapter does not expose the requested site or operation."""


class HookExecutionError(EEGLensError):
    """A native module did not execute the declared number of times."""
