import sys
import types
import warnings

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message="pkg_resources is deprecated"
)


def ensure_pkg_resources():
    if "pkg_resources" in sys.modules:
        return sys.modules["pkg_resources"]

    try:
        import pkg_resources

        return pkg_resources
    except ImportError:
        try:
            from setuptools.extern import pkg_resources as pr

            sys.modules["pkg_resources"] = pr
            return pr
        except ImportError:
            from importlib.metadata import version as get_version

            pr = types.ModuleType("pkg_resources")
            pr.get_distribution = lambda name: types.SimpleNamespace(
                project_name=name, version=get_version(name)
            )
            pr.Distribution = type(
                "Distribution",
                (),
                {
                    "__init__": lambda self, name, version: (
                        setattr(self, "project_name", name)
                        or setattr(self, "version", version)
                    )
                },
            )
            sys.modules["pkg_resources"] = pr
            return pr


pkg_resources = ensure_pkg_resources()
