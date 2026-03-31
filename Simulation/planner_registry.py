import importlib
import inspect

from config import DEFAULT_GLOBAL_PLANNER, DEFAULT_LOCAL_PLANNER
from planner_interface import GlobalPlannerInterface, LocalPlannerInterface


def _normalize_module_name(name):
    if not name:
        return None
    return name[:-3] if name.endswith(".py") else name


def _load_planner_class(module_name, base_class):
    module = importlib.import_module(module_name)

    if hasattr(module, "Planner"):
        planner_class = module.Planner
    else:
        candidates = [
            obj
            for obj in module.__dict__.values()
            if inspect.isclass(obj) and issubclass(obj, base_class) and obj is not base_class
        ]
        if len(candidates) != 1:
            raise ImportError(
                f"Planner module '{module_name}' must define Planner or exactly one {base_class.__name__} subclass "
                f"(found {len(candidates)})."
            )
        planner_class = candidates[0]

    if not issubclass(planner_class, base_class):
        raise TypeError(f"Planner in '{module_name}' does not implement {base_class.__name__}.")

    return planner_class


def load_global_planner(planner_name=None):
    module_name = _normalize_module_name(planner_name or DEFAULT_GLOBAL_PLANNER)
    if not module_name:
        return None
    planner_class = _load_planner_class(module_name, GlobalPlannerInterface)
    return planner_class()


def load_local_planner(planner_name=None):
    module_name = _normalize_module_name(planner_name or DEFAULT_LOCAL_PLANNER)
    if not module_name:
        return None
    planner_class = _load_planner_class(module_name, LocalPlannerInterface)
    return planner_class()
