from typing import Callable, List, Union


def depends_on(required_tasks: Union[str, List[str]]) -> Callable:
    """Decorator to mark required predecessor analysis functions."""

    # Ensure it's always a list internally
    if isinstance(required_tasks, str):
        required_tasks = [required_tasks]

    def decorator(func: Callable) -> Callable:
        # Attach the list of dependencies directly to the function object
        setattr(func, "_pipeline_dependencies", required_tasks)
        return func

    return decorator
