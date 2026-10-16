"""Local deployment: inference telemetry and remote image uploads disabled."""
def log_inference(*args, **kwargs):
    return None
