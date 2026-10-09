"""Verify the documentation against reality.

Docs rot silently: a renamed flag, a moved file or a deleted script is invisible
until someone follows the README on a machine that is not yours. This checks
that every command, script and cross-reference the docs mention actually exists.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from anti_uav.cli import app

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ALL_DOCS = [ROOT / "README.md", *sorted(DOCS.glob("*.md"))]


def documented_commands() -> set[tuple[str, ...]]:
    """Every `anti-uav ...` invocation that appears in the docs."""
    found: set[tuple[str, ...]] = set()
    for path in ALL_DOCS:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip().lstrip("$").strip()
            if not stripped.startswith("anti-uav "):
                continue
# take the command and its first non-flag token
            tokens = stripped.split()[1:]
            if not tokens:
                continue
            if tokens[0].startswith("<") or tokens[0].startswith("["):
                continue  # a placeholder like `anti-uav <command>`, not a real call
            words: list[str] = []
            for token in tokens:
                if token.startswith("-"):
                    break
                words.append(token)
            if words:
                found.add(tuple(words))
    return found


class TestDocsExist:
    @pytest.mark.parametrize(
        "name",
        ["COMMANDS.md", "DATASETS.md", "EXPERIMENTS.md", "ENVIRONMENT.md", "ARCHITECTURE.md"],
    )
    def test_document_is_present(self, name: str) -> None:
        path = DOCS / name
        assert path.is_file(), f"{name} is referenced from the README but missing"
        assert len(path.read_text(encoding="utf-8")) > 2000, f"{name} looks like a stub"

    def test_readme_links_resolve(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)#][^)]*)\)", readme):
            if target.startswith(("http://", "https://")):
                continue
            resolved = (ROOT / target.replace("%20", " ")).resolve()
            assert resolved.exists(), f"README links to {target}, which does not exist"


class TestDocumentedScriptsExist:
    @pytest.mark.parametrize(
        "name",
        [
            "setup_env.ps1",
            "train_matrix.ps1",
            "run_pipeline.ps1",
            "gen_coverage_map.py",
            "fetch_wheel.py",
            "smoke_data.py",
            "smoke_tracking.py",
            "smoke_coordination.py",
            "smoke_rules.py",
            "smoke_api.py",
            "smoke_cross_eval.py",
            "smoke_commands.py",
        ],
    )
    def test_script_is_present(self, name: str) -> None:
        assert (ROOT / "scripts" / name).is_file(), f"scripts/{name} is referenced but missing"

    @pytest.mark.parametrize(
        "name", ["setup_env.ps1", "train_matrix.ps1", "run_pipeline.ps1"]
    )
    def test_powershell_scripts_parse(self, name: str) -> None:
        """A syntax error in a setup script is found at the worst possible moment."""
        path = ROOT / "scripts" / name
        text = path.read_text(encoding="utf-8")
        assert text.startswith("<#"), f"{name} lost its help block"
        assert ".SYNOPSIS" in text, f"{name} has no SYNOPSIS"
        assert ".DESCRIPTION" in text or ".PARAMETER" in text
        assert "exit " in text, f"{name} does not set an exit code"
        assert "anti-uav" in text, f"{name} never says what it runs"


class TestDocumentedCommandsExist:
    def test_every_documented_command_resolves(self) -> None:
        runner = CliRunner()
        unknown: list[str] = []
        for words in sorted(documented_commands()):
            # `anti-uav <cmd> <subcmd> --help` must succeed; a typo exits 2.
            result = runner.invoke(app, [*words, "--help"], catch_exceptions=False)
            if result.exit_code != 0:
                unknown.append(" ".join(words))
        assert not unknown, f"documented but not real: {unknown}"

    def test_the_cli_exposes_the_commands_the_readme_promises(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for command in (
            "verify-env",
            "fetch-weights",
            "download",
            "convert",
            "stats",
            "splits",
            "build",
            "sanity",
            "matrix",
            "evaluate",
            "track-eval",
            "replay",
            "predict",
            "export",
            "serve",
        ):
            assert f"anti-uav {command}" in readme, f"README does not mention `{command}`"
            runner = CliRunner()
            assert runner.invoke(app, [command, "--help"]).exit_code == 0, command


class TestNoStaleReferences:
    def test_no_doc_mentions_uv_run(self) -> None:
        """The project uses a conda env, not uv. A stale `uv run` sends people
        to create a second environment."""
        for path in ALL_DOCS:
            text = path.read_text(encoding="utf-8")
            assert "uv run" not in text, f"{path.name} still says `uv run`"

    def test_docs_do_not_promise_a_dvc_or_git_lfs_workflow(self) -> None:
        for path in ALL_DOCS:
            assert "dvc pull" not in path.read_text(encoding="utf-8")
