# Demo Fallback Mode

RetinaGuard-AI includes a demo fallback mode for interface testing.

If real model inference fails because checkpoints or configuration files are missing, the API returns a structured demo result instead of crashing.

This mode is useful for:

- Testing the FastAPI backend.
- Capturing web interface screenshots.
- Demonstrating the UI flow on GitHub.

Important:

- Demo fallback outputs are not real medical predictions.
- Demo fallback outputs are not suitable for diagnosis.
- Real inference requires trained local checkpoints and correct API configuration.

The web interface labels fallback results as Demo Mode when this path is used.
