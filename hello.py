"""Minimal CML Model smoke test — no dependencies, no data, just arithmetic.

Used to isolate whether CML Model builds/deployments work at all in this
environment, independent of our forecasting code's dependencies.
"""


def add_numbers(args):
    """CML Model entry point.

    Parameters
    ----------
    args : dict
        Expected shape: {"a": <number>, "b": <number>}

    Returns
    -------
    dict
        {"sum": <number>} on success, or {"sum": None, "error": "..."} on
        bad/missing input (must not raise — CML self-tests with a
        placeholder payload during build).
    """
    if not isinstance(args, dict):
        return {"sum": None, "error": "args must be a dict"}

    a = args.get("a")
    b = args.get("b")
    if a is None or b is None:
        return {"sum": None, "error": "missing 'a' or 'b' in request"}

    try:
        return {"sum": a + b}
    except Exception as e:
        return {"sum": None, "error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    print(add_numbers({}))
    print(add_numbers({"a": 2, "b": 3}))
    print(add_numbers({"a": "x", "b": 3}))
