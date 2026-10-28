"""Source-tree resource/entrypoint contracts, NOT wheel or fresh-install acceptance.

These checks use the existing interpreter and repository only. They neither build nor
install a distribution, contact a package index, or inspect local runtime/model data.
"""
from __future__ import annotations

import ast
import importlib
import re
import tomllib
from html.parser import HTMLParser
from importlib.resources import files
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "src" / "industrial_diagnosis"


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids = []
        self.assets = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "script" and "src" in attrs:
            self.assets.append(attrs["src"])
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.assets.append(attrs["href"])


@pytest.mark.parametrize("module", [
    "__init__", "api", "cli", "fixtures", "knowledge", "managed_provider",
    "provider", "reports", "schemas", "service", "storage", "telemetry",
])
def test_runtime_module_import_is_from_this_source_tree(module):
    name = "industrial_diagnosis" + ("." + module if module != "__init__" else "")
    imported = importlib.import_module(name)
    assert Path(imported.__file__).resolve() == (SOURCE / f"{module}.py").resolve()


@pytest.mark.parametrize("resource", ["templates/index.html", "static/app.js", "static/style.css"])
def test_importlib_resources_resolves_source_assets(resource):
    expected = SOURCE / resource
    resolved = files("industrial_diagnosis").joinpath(*resource.split("/"))
    assert expected.is_file() and not expected.is_symlink()
    assert resolved.read_bytes() == expected.read_bytes()
    assert resolved.read_text(encoding="utf-8").strip()


def test_declared_console_entrypoint_resolves_to_source_callable():
    metadata = tomllib.loads((PROJECT / "pyproject.toml").read_text())
    entry = metadata["project"]["scripts"]["industrial-diagnosis"]
    assert entry == "industrial_diagnosis.cli:main"
    module, symbol = entry.split(":")
    imported = importlib.import_module(module)
    assert Path(imported.__file__).resolve() == (SOURCE / "cli.py").resolve()
    assert callable(getattr(imported, symbol))
    tree = ast.parse((SOURCE / "cli.py").read_text())
    assert any(isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
               and ast.unparse(node.test) == "__name__ == '__main__'"
               and any(isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                       and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "main"
                       for stmt in node.body) for node in tree.body)


def test_wheel_configuration_declares_the_resource_owning_source_package():
    metadata = tomllib.loads((PROJECT / "pyproject.toml").read_text())
    assert metadata["build-system"]["build-backend"] == "hatchling.build"
    wheel = metadata["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["packages"] == ["src/industrial_diagnosis"]
    # This validates configuration, not a built archive or a new environment.
    assert (SOURCE / "__init__.py").is_file()


def test_page_references_only_the_shipped_local_script_and_stylesheet():
    page = Page((SOURCE / "templates/index.html").read_text())
    assert sorted(page.assets) == ["/static/app.js", "/static/style.css"]
    for asset in page.assets:
        assert (SOURCE / asset.lstrip("/")).is_file()


def test_javascript_dom_bindings_resolve_to_unique_source_page_ids():
    page = Page((SOURCE / "templates/index.html").read_text())
    assert len(page.ids) == len(set(page.ids))
    script = (SOURCE / "static/app.js").read_text()
    referenced = set(re.findall(r'\b(?:el|on|show|choices)\(\s*["\']([^"\']+)["\']', script))
    assert referenced
    assert referenced <= set(page.ids), f"Missing source HTML IDs: {referenced - set(page.ids)}"


def test_cli_provider_selection_has_no_implicit_model_or_test_fallback():
    from industrial_diagnosis.cli import provider_for
    from industrial_diagnosis.provider import SyntheticProvider

    provider = provider_for(None, None, True)
    assert type(provider) is SyntheticProvider and provider.is_test_provider is True
    import typer

    with pytest.raises(typer.BadParameter):
        provider_for(None, None, False)
    with pytest.raises(typer.BadParameter):
        provider_for(None, "http://127.0.0.1:9", True)
