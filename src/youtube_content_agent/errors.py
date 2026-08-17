class AgentError(RuntimeError):
    """Base error with a user-actionable message."""


class ExternalToolError(AgentError):
    """An external CLI or API failed."""


class TranscriptUnavailableError(AgentError):
    """No usable timestamped transcript was available."""


class GroundingError(AgentError):
    """Generated editorial data cannot be traced to the transcript."""


class ConfigurationError(AgentError):
    """Required production configuration is absent or invalid."""
