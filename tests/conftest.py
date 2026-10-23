"""Test configuration and fixtures for DeadlockTutorLLM test suite.

Ensures the test suite runs deterministically with the mock LLM provider,
independent of local model daemon or external environment state.
"""

import os

# Set test environment defaults before any application modules are imported
os.environ["LLM_PROVIDER"] = "mock"
os.environ["ENVIRONMENT"] = "testing"
