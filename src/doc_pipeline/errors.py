class PipelineError(Exception):
    """An expected failure that should send the source PDF to errors/."""


class ConfigurationError(PipelineError):
    """Invalid or incomplete local configuration."""


class BoundaryDetectionError(PipelineError):
    """The boundary model returned invalid or uncertain page classifications."""


class ProviderError(PipelineError):
    """A provider request failed or returned unusable structured output."""


class RunStopped(PipelineError):
    """The run hit its time limit or was cancelled."""
