#!/usr/bin/env python3
"""Print elevation_node's ROS parameters -- defaults and what each one does.

WHY a source parser instead of `ros2 param describe`: the node declares ~120 parameters with no
ParameterDescriptor text, so `ros2 param` can only ever show you a name and a value. The real
documentation is the WHY comments wrapped around each declaration in elevation_node.py. Reading
those straight out of the source means this reference can never drift from the code, and it works
with no node running.

Usage:
    ros/params.py                 # every parameter, grouped exactly as the source groups them
    ros/params.py plan            # only parameters (or groups) matching a substring
    ros/params.py speed wmax      # several needles -> match any
"""
from __future__ import annotations

import re
import shutil
import sys
import textwrap
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

NODE = Path(__file__).resolve().parent / "helhest_stack_ros/helhest_stack_ros/elevation_node.py"

# `d("name", default)` with an optional trailing `# note`. The non-greedy default stops at the
# first `)`, which is enough: no declared default contains one.
_DECL = re.compile(r'^d\("(?P<name>\w+)",\s*(?P<default>.+?)\)(?:\s*#\s*(?P<note>.*))?$')
# The node's construction-time parameter sets, e.g. `_PLAN_BUILD = frozenset({...})`.
_BUILD_SET = re.compile(r"^_(\w+)_BUILD = frozenset\(\s*\{(.*?)\}\s*\)", re.MULTILINE | re.DOTALL)
# A comment hanging-indented under a trailing note continues that note (see _parse).
_HANGING = re.compile(r"#\s{4,}\S")

# What each *_BUILD set rebuilds, for the tag next to a parameter.
_BUILDS = {
    "ICP": "ICP aligner",
    "ACC": "accumulator",
    "DYN": "dynamic filter",
    "OUTLIER": "outlier filter",
    "PLAN": "planner",
}


@dataclass
class Param:
    name: str
    default: str
    note: str = ""
    rebuilds: list[str] = field(default_factory=list)


@dataclass
class Group:
    """A comment block plus every parameter declared under it -- the source's own grouping."""

    doc: str
    params: list[Param] = field(default_factory=list)


def _body(src: str) -> list[str]:
    """The lines inside `def _declare_parameters`, up to the next method."""
    lines = src.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith("def _declare_parameters"))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("    def "))
    return lines[start + 1 : end]


def _rebuild_map(src: str) -> dict[str, list[str]]:
    """name -> the objects a change to it rebuilds, read from the _*_BUILD frozensets."""
    out: dict[str, list[str]] = {}
    for kind, members in _BUILD_SET.findall(src):
        label = _BUILDS.get(kind, kind.lower())
        for name in re.findall(r'"(\w+)"', members):
            out.setdefault(name, []).append(label)
    return out


def _parse(src: str) -> list[Group]:
    groups: list[Group] = [Group(doc="")]
    pending: list[str] = []  # comment lines gathered for the group about to open
    last: Param | None = None  # most recent parameter, for continuation notes
    after_decl = False  # previous line was a declaration
    cont: Param | None = None  # parameter whose note the current comment block continues

    for raw in _body(src):
        line = raw.strip()
        if not line:
            after_decl, cont = False, None
            continue

        if line.startswith("#"):
            text = line.lstrip("#").strip()
            # A comment block normally opens a group and documents what follows it. Two blocks in
            # the source instead CONTINUE the note of the parameter above: they either pick up a
            # note left open with ';' or hang indented under it. Attribute those to that parameter
            # so its text does not land on the next one.
            if cont is None and after_decl and last is not None:
                if last.note.endswith(";") or _HANGING.match(line):
                    cont = last
            if cont is not None:
                cont.note = f"{cont.note} {text}".strip()
            else:
                pending.append(text)
            after_decl = False
            continue

        m = _DECL.match(line)
        if not m:
            after_decl, cont = False, None
            continue

        if pending:
            groups.append(Group(doc=" ".join(pending)))
            pending = []
        last = Param(m["name"], m["default"], (m["note"] or "").strip())
        groups[-1].params.append(last)
        after_decl, cont = True, None

    return [g for g in groups if g.params]


def _select(groups: list[Group], needles: list[str]) -> list[Group]:
    """Groups matching a needle keep every parameter; otherwise only the matching parameters."""
    if not needles:
        return groups
    hits: list[Group] = []
    for g in groups:
        if any(n in g.doc.lower() for n in needles):
            hits.append(g)
            continue
        keep = [p for p in g.params if any(n in f"{p.name} {p.note}".lower() for n in needles)]
        if keep:
            hits.append(Group(doc=g.doc, params=keep))
    return hits


def _print(groups: list[Group]) -> None:
    color = sys.stdout.isatty()
    dim, bold, cyan, off = ("\033[2m", "\033[1m", "\033[36m", "\033[0m") if color else ("",) * 4
    width = min(shutil.get_terminal_size((100, 24)).columns, 100)

    for g in groups:
        print()
        if g.doc:
            for line in textwrap.wrap(g.doc, width=width - 2):
                print(f"{dim}# {line}{off}")
        for p in g.params:
            tag = f"  {cyan}[rebuilds {', '.join(p.rebuilds)}]{off}" if p.rebuilds else ""
            print(f"  {bold}{p.name}{off} = {p.default}{tag}")
            for line in textwrap.wrap(p.note, width=width - 6):
                print(f"      {dim}{line}{off}")


def main() -> int:
    src = NODE.read_text()
    groups = _parse(src)
    rebuilds = _rebuild_map(src)
    for g in groups:
        for p in g.params:
            p.rebuilds = rebuilds.get(p.name, [])

    needles = [a.lower() for a in sys.argv[1:]]
    hits = _select(groups, needles)
    if not hits:
        print(f"no parameter matching {' '.join(needles)}", file=sys.stderr)
        return 1

    _print(hits)
    count = sum(len(g.params) for g in hits)
    print(f"\n{count} parameters. Set one at launch:")
    print("  ros2 run helhest_stack_ros elevation_node --ros-args -p goal_source:=follow")
    print("...or live on the running node (a [rebuilds ...] one costs a rebuild, but still works):")
    print("  ros2 param set /elevation_node follow_standoff 2.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
