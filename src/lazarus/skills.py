"""Discover skill metadata once; leave instruction loading to the model."""

from dataclasses import dataclass
from html import escape
import os
from pathlib import Path
import re

import yaml


MAX_HEADER_CHARS = 64 * 1024
MAX_CATALOG_CHARS = 16 * 1024
MAX_DIRECTORIES = 2_000
SKILL_GUIDANCE = """The following skills provide specialized instructions.
When a task matches a description, read that skill's SKILL.md through Python
before acting. Follow its instructions and load required references. Read each
selected instruction file completely; if output is truncated, continue in
chunks until EOF. Progressive disclosure means choosing relevant files, not
silently cutting off their instructions. Resolve relative paths from the skill
directory. Prefer provided scripts and helpers; creatively compose them through
Python. Reuse instructions already in context. After a new loop, reload any
needed instructions that were not preserved in the handoff. If a skill is
missing or cannot be used, explain briefly and use the best available fallback.
"""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    disable_model_invocation: bool = False


def skill_roots(cwd: Path, home: Path) -> list[Path]:
    """Nearest project wins; stop at the Git root, then include global skills."""
    roots: list[Path] = []
    cwd = cwd.resolve()
    for directory in (cwd, *cwd.parents):
        roots.append(directory / ".agents/skills")
        if (directory / ".git").exists():
            break
    roots.append(home / ".agents/skills")
    return list(dict.fromkeys(roots))


def _read_skill(path: Path) -> Skill:
    with path.open(encoding="utf-8-sig") as stream:
        lines = stream.read(MAX_HEADER_CHARS).splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing YAML frontmatter")
    end = next(
        (i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"), None
    )
    if end is None:
        raise ValueError("unterminated or oversized YAML frontmatter")
    metadata = yaml.safe_load("\n".join(lines[1:end]))
    if not isinstance(metadata, dict):
        raise ValueError("frontmatter must be a mapping")
    name = metadata.get("name")
    description = metadata.get("description")
    if (
        not isinstance(name, str)
        or len(name) > 64
        or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
    ):
        raise ValueError("name must be 1–64 lowercase letters, digits, or hyphens")
    if (
        not isinstance(description, str)
        or not description.strip()
        or len(description) > 1024
    ):
        raise ValueError("description must be 1–1024 characters")
    disabled = metadata.get("disable-model-invocation", False)
    if not isinstance(disabled, bool):
        raise ValueError("disable-model-invocation must be a boolean")
    return Skill(name, " ".join(description.split()), path.resolve(), disabled)


def discover_skills(
    cwd: Path, home: Path | None = None
) -> tuple[list[Skill], list[str]]:
    skills: dict[str, Skill] = {}
    warnings: list[str] = []
    seen_files: set[Path] = set()
    seen_dirs: set[Path] = set()

    def on_error(error: OSError) -> None:
        warnings.append(str(error))

    for root in skill_roots(cwd, home or Path.home()):
        if not root.is_dir():
            continue
        for directory, dirs, files in os.walk(root, followlinks=True, onerror=on_error):
            path = Path(directory)
            try:
                real = path.resolve()
                if real in seen_dirs:
                    dirs.clear()
                    continue
                if len(seen_dirs) >= MAX_DIRECTORIES:
                    warnings.append(
                        "Skill directory scan limit reached; some skills omitted."
                    )
                    return list(skills.values()), warnings
                seen_dirs.add(real)
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                if "SKILL.md" not in files:
                    continue
                # References and assets inside a skill are not separate skills.
                dirs.clear()
                manifest = path / "SKILL.md"
                resolved = manifest.resolve()
                if resolved in seen_files or not manifest.is_file():
                    continue
                seen_files.add(resolved)
                skill = _read_skill(manifest)
                if skill.name in skills:
                    warnings.append(
                        f"Duplicate skill {skill.name}: using {skills[skill.name].path}; "
                        f"skipping {manifest}."
                    )
                else:
                    skills[skill.name] = skill
            except (OSError, ValueError, RuntimeError, yaml.YAMLError) as error:
                warnings.append(f"Skipping skill at {path}: {error}")
    return list(skills.values()), warnings


def skills_prompt(cwd: Path) -> tuple[str, list[str]]:
    skills, warnings = discover_skills(cwd)
    entries: list[str] = []
    size = 0
    for skill in skills:
        if skill.disable_model_invocation:
            continue
        entry = (
            f"  <skill name={escape_attr(skill.name)} "
            f"path={escape_attr(str(skill.path))}>"
            f"{escape(skill.description)}</skill>"
        )
        if size + len(entry) > MAX_CATALOG_CHARS:
            warnings.append(f"Skill catalog budget reached; omitted {skill.name}.")
            continue
        entries.append(entry)
        size += len(entry) + 1
    if not entries:
        return "", warnings
    return (
        "\n\n"
        + SKILL_GUIDANCE
        + "\n<available_skills>\n"
        + "\n".join(entries)
        + "\n</available_skills>",
        warnings,
    )


def escape_attr(value: str) -> str:
    return '"' + escape(value, quote=True) + '"'
