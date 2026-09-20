"""Environment namespace without import-time backend initialization.

HKBZ data utilities do not require simulator-specific flags or training
dependencies. Optional backends must initialize their own command-line flags
at their explicit entrypoints, not when this package is imported.
"""
