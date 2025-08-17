import os

# Import HyperbolicClient if available; otherwise, set to None to avoid import errors during testing.
try:
    from hyperbolic import HyperbolicClient  # type: ignore
except Exception:
    HyperbolicClient = None  # type: ignore


def connect_to_hyperbolic(api_key: str, endpoint: str):
    """Initialize a connection to the Hyperbolic API and return available models to verify connection."""
    if HyperbolicClient is None:
        raise ImportError("HyperbolicClient is not available. Install the hyperbolic package.")
    client = HyperbolicClient(api_key=api_key, endpoint=endpoint)
    try:
        # Example API call: list available models
        models = client.list_models()
        return models
    except Exception as e:
        print(f"Error connecting to Hyperbolic: {e}")
        return None


if __name__ == '__main__':
    api_key = os.getenv('HYPERBOLIC_API_KEY')
    endpoint = os.getenv('HYPERBOLIC_ENDPOINT', 'https://api.hyperbolic.xyz')

    if not api_key:
        print("Error: HYPERBOLIC_API_KEY environment variable is not set.")
        exit(1)

    models = connect_to_hyperbolic(api_key, endpoint)
    if models is not None:
        print("Successfully connected to Hyperbolic. Available models:", models)
    else:
        print("Failed to retrieve models from Hyperbolic.")
