#!/usr/bin/env python3
"""Check source, local imports, documentation links, and repository hygiene.

Uses only the standard library; does not import or execute model code.
Run from any directory: python /path/to/Agima/tools/check_repo.py
"""

from __future__ import annotations

import ast
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
# Archives have no Git index or ignore processing. Keep downloaded assets and
# generated runs out of their source walk, matching the documented local paths.
LOCAL_ARTIFACT_DIRS = {
    "models", "pretrained", "ckpts", "ckpts_snapshot", "datasets", "hf_datasets",
    "results", "results_prev", "runs", "runs_understanding", "outputs", "artifacts",
    "logs", "wandb", "eval_results", "data/bagel_example",
    "experiments/image_ablation/runs", "experiments/image_ablation/output",
    "experiments/image_ablation/prepared", "experiments/image_ablation/assets",
}
REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "requirements.txt",
    ".env.example",
    "scripts/train.py",
    "scripts/evaluate.py",
)
MACHINE_PATH = re.compile(
    r"(?<![\w./])/(?:data/home|storage/home|home|Users|scratch[0-9]*|checkpoint(?:/forge)?|mnt|data/shared|datasets)/[^\s/'\"]+"
)
# Optional organization-specific identifiers are supplied at check time; private
# account lists are not stored in the public repository.
CONFLICT_MARKER = re.compile(r"^(?:<{7}|>{7})(?: |$)", re.MULTILINE)
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".pt", ".pth", ".safetensors", ".ckpt"}
BINARY_SUFFIXES = FORBIDDEN_SUFFIXES | {
    ".bin", ".npy", ".npz", ".parquet", ".arrow", ".gz", ".zip", ".tar",
    ".png", ".gif", ".jpg", ".jpeg", ".webp", ".bmp", ".ico", ".pdf",
    ".ttf", ".otf", ".woff", ".woff2", ".mp4", ".webm", ".wav", ".mp3",
    ".so", ".dylib", ".dll",
}
SOKOBAN_SPRITES = (
    "box", "box_on_target", "box_target", "floor", "player", "player_on_target", "wall",
)
LOCAL_ROOTS = {"data", "evaluation", "experiments", "gymnasium", "modeling", "scripts", "training", "tools", "inferencer"}
REMOVED_ROOTS = {"eval", "rl", "train", "visgym_training", "gen_eval_wrappers", "visualize_gifs", "utils"}
LINK_DESTINATION = r"(?:<[^>\n]+>|(?:\\.|[^\s()]|\([^()\n]*\))+)"
INLINE_LINK = re.compile(r"\[[^\]\n]*\]\(\s*(" + LINK_DESTINATION + r")(?:\s+[\"'][^\n]*?[\"'])?\s*\)")
LINK_DEFINITION = re.compile(r"^\s{0,3}\[([^\]]+)\]:\s*(" + LINK_DESTINATION + r")", re.MULTILINE)


def markdown_sections(text: str):
    """Separate prose and fenced examples, preserving source line numbers."""
    text = re.sub(r"<!--[\s\S]*?-->", lambda match: "\n" * match.group().count("\n"), text)
    prose, blocks = [], []
    fence = None
    start, language, body = 0, "", []
    for number, line in enumerate(text.splitlines(), 1):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                blocks.append((start, language, "\n".join(body)))
                fence = None
            else:
                body.append(line)
            prose.append("")
        elif marker:
            fence, language, start, body = marker[1], marker[2].strip().lower(), number + 1, []
            prose.append("")
        else:
            prose.append(line)
    return "\n".join(prose), blocks


def markdown_anchors(text: str) -> set[str]:
    prose, _ = markdown_sections(text)
    anchors = set(re.findall(r"\b(?:id|name)=[\"']([^\"']+)[\"']", prose))
    lines = prose.splitlines()
    for index, line in enumerate(lines):
        atx = re.match(r"^ {0,3}#{1,6}\s+(.+?)(?:\s+#+)?\s*$", line)
        heading = atx[1] if atx else None
        if not heading and index and re.fullmatch(r" {0,3}(?:=+|-+)\s*", line) and lines[index - 1].strip():
            heading = lines[index - 1]
        if heading is None:
            continue
        heading = re.sub(r"!?\[([^\]]+)\]\([^)]*\)", r"\1", heading)
        heading = html.unescape(re.sub(r"<[^>]*>", "", heading)).lower()
        slug = re.sub(r"[^\w\- ]", "", heading).replace(" ", "-")
        anchor, suffix = slug, 0
        while anchor in anchors:
            suffix += 1
            anchor = f"{slug}-{suffix}"
        anchors.add(anchor)
    return anchors


def markdown_links(text: str):
    """Read inline/reference links and quoted HTML href/src, outside code."""
    prose, _ = markdown_sections(text)
    prose = re.sub(r"(`+).*?\1", lambda match: " " * len(match.group()), prose)
    definitions = {" ".join(match[1].lower().split()): match[2]
                   for match in LINK_DEFINITION.finditer(prose)}
    for number, line in enumerate(prose.splitlines(), 1):
        definition = LINK_DEFINITION.match(line)
        if definition:
            yield number, definition[2]
            continue
        for match in INLINE_LINK.finditer(line):
            yield number, match[1]
        for match in re.finditer(r"\[([^\]]+)\]\[([^\]]*)\]", line):
            label = " ".join((match[2] or match[1]).lower().split())
            yield number, definitions.get(label)  # None denotes an undefined reference.
        for match in re.finditer(r"(?<!!)\[([^\]]+)\](?![\[(])", line):
            label = " ".join(match[1].lower().split())
            if label in definitions:
                yield number, definitions[label]
        for match in re.finditer(r"\b(?:href|src)=[\"']([^\"']+)[\"']", line):
            yield number, match[1]


def check_markdown_links(root: Path) -> tuple[int, list[str]]:
    """Check maintained Markdown locally; external URLs are not fetched."""
    root = root.resolve()
    errors, count = [], 0
    for path in repository_files(root):
        relative = path.relative_to(root)
        if path.suffix.lower() != ".md" or relative.parts[0] in {"gymnasium", "LICENSES"}:
            continue  # Vendored attribution is preserved verbatim.
        count += 1
        for number, destination in markdown_links(path.read_text(encoding="utf-8")):
            location = f"{relative}:{number}"
            if destination is None:
                errors.append(f"{location}: undefined Markdown link reference")
                continue
            destination = html.unescape(destination.strip("<>"))
            url = urlsplit(destination)
            if url.scheme or url.netloc or destination.startswith("/"):
                continue
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if not target.is_relative_to(root) or not target.exists():
                errors.append(f"{location}: missing local link target {destination}")
                continue
            if url.fragment and target.suffix.lower() == ".md":
                if unquote(url.fragment) not in markdown_anchors(target.read_text(encoding="utf-8")):
                    errors.append(f"{location}: missing Markdown heading {destination}")
    return count, sorted(set(errors))


def readme_python_commands(root: Path):
    """Read shell-fenced Python commands, including continued lines; never run them."""
    readme = root / "README.md"
    if not readme.is_file():
        return
    _, blocks = markdown_sections(readme.read_text(encoding="utf-8"))
    for start, language, body in blocks:
        if language not in {"bash", "sh", "shell", "console"}:
            continue
        pending, first = "", start
        for offset, line in enumerate(body.splitlines()):
            if not pending:
                first = start + offset
            pending += line.removesuffix("\\") + (" " if line.endswith("\\") else "")
            if line.endswith("\\"):
                continue
            try:
                words = shlex.split(pending.removeprefix("$ "), comments=True)
            except ValueError:
                if re.search(r"\bpython(?:3(?:\.\d+)?)?\b", pending):
                    yield first, []
                pending = ""
                continue
            pending = ""
            while words and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]):
                words.pop(0)
            if words and re.fullmatch(r"python(?:3(?:\.\d+)?)?", words[0]):
                yield first, words
        if pending and re.search(r"\bpython(?:3(?:\.\d+)?)?\b", pending):
            yield first, []


def check_readme_commands(root: Path) -> tuple[int, list[str]]:
    """Resolve README Python entrypoints and check local argparse option names."""
    errors, count = [], 0
    for number, command in readme_python_commands(root):
        count += 1
        if not command:
            errors.append(f"README.md:{number}: malformed shell command")
            continue
        args = command[1:]
        if not args or args[0] == "-c":
            continue
        if args[0] == "-m":
            if len(args) < 2:
                errors.append(f"README.md:{number}: missing Python module")
                continue
            module, args = args[1], args[2:]
            if module.split(".")[0] in sys.stdlib_module_names or module == "pip":
                continue
            source = root.joinpath(*module.split(".")).with_suffix(".py")
            if not source.is_file():
                source = root.joinpath(*module.split("."), "__main__.py")
        else:
            source, args = root / args[0], args[1:]
        if not source.is_file():
            errors.append(f"README.md:{number}: missing Python entrypoint {command[1]}")
            continue
        try:
            tree = ast.parse(source.read_bytes())
        except (OSError, SyntaxError, UnicodeError, ValueError) as exc:
            errors.append(f"README.md:{number}: cannot parse Python entrypoint {command[1]}: {exc}")
            continue
        options = {arg.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                   for arg in node.args if isinstance(arg, ast.Constant)
                   and isinstance(arg.value, str) and arg.value.startswith("-")}
        if options:
            unknown = {arg.split("=", 1)[0] for arg in args if arg.startswith("--")} - options - {"--help"}
            for option in sorted(unknown):
                errors.append(f"README.md:{number}: unknown option {option} for {source.relative_to(root)}")
    return count, errors


def has_repository_index(root: Path) -> bool:
    """A source archive must not inherit an unrelated parent repository's ignores."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True, check=True, text=True,
        )
        return Path(result.stdout.strip()).resolve() == root.resolve()
    except (OSError, subprocess.CalledProcessError):
        return False


def repository_files(root: Path):
    """Include tracked and new source, but omit ignored local downloads/caches."""
    try:
        if not has_repository_index(root):
            raise OSError("Source directory has no repository index of its own")
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True, check=True,
        )
        paths = {root / os.fsdecode(name) for name in result.stdout.split(b"\0") if name}
        yield from sorted(path for path in paths if path.is_file())
        return
    except (OSError, subprocess.CalledProcessError):
        pass
    # Source archives have no Git index.
    for parent, directories, names in os.walk(root):
        relative = Path(parent).relative_to(root)
        directories[:] = sorted(d for d in directories if d not in SKIP_DIRS
                                and (relative / d).as_posix() not in LOCAL_ARTIFACT_DIRS)
        for name in sorted(names):
            yield Path(parent) / name


def source_files(root: Path):
    yield from (path for path in repository_files(root) if path.suffix == ".py")


def check_python(root: Path) -> tuple[int, list[str]]:
    errors = []
    count = 0
    for path in source_files(root):
        count += 1
        relative = path.relative_to(root)
        try:
            # Bytes preserve Python's declared source encoding. No imports, GPU
            # initialization, or __pycache__ writes occur here.
            ast.parse(path.read_bytes(), filename=str(relative))
        except (OSError, SyntaxError, UnicodeError, ValueError) as exc:
            errors.append(f"{relative}: {exc}")
    return count, errors


def check_public_files(root: Path) -> list[str]:
    return [f"Missing repository file: {name}" for name in REQUIRED_FILES
            if not (root / name).is_file()]


def check_license_copies(root: Path) -> tuple[int, list[str]]:
    """Verify vendored license bytes against the recorded provenance manifest."""
    root = root.resolve()
    directory = (root / "LICENSES").resolve()
    manifest = directory / "sources.json"
    try:
        sources = json.loads(manifest.read_text(encoding="utf-8"))["sources"]
        if not isinstance(sources, list) or not sources:
            raise ValueError("sources must be a nonempty list")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return 0, [f"LICENSES/sources.json: invalid license manifest: {exc}"]
    errors, recorded = [], set()
    for index, source in enumerate(sources, 1):
        location = f"LICENSES/sources.json: entry {index}"
        if not isinstance(source, dict):
            errors.append(f"{location}: expected an object")
            continue
        name, digest, size = (source.get(key) for key in ("file", "sha256", "bytes"))
        if (not isinstance(name, str) or not name or Path(name).is_absolute()
                or not (directory / name).resolve().is_relative_to(directory)):
            errors.append(f"{location}: file must remain inside LICENSES")
            continue
        path = (directory / name).resolve()
        if path in recorded:
            errors.append(f"{location}: duplicate license file {name}")
        recorded.add(path)
        if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or type(size) is not int or size < 1):
            errors.append(f"{location}: invalid sha256 or bytes for {name}")
            continue
        try:
            content = path.read_bytes()
        except OSError as exc:
            errors.append(f"LICENSES/{name}: cannot read license copy: {exc}")
            continue
        if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
            errors.append(f"LICENSES/{name}: license copy differs from its recorded bytes/SHA-256")
    for path in directory.rglob("*.txt"):
        if path.resolve() not in recorded:
            errors.append(f"{path.relative_to(root)}: license copy is missing from sources.json")
    return len(recorded), errors


def check_config_references(root: Path) -> tuple[int, list[str]]:
    """Resolve explicit Python recipe paths; this does not parse YAML schemas."""
    root = root.resolve()
    errors, references = [], set()
    for path in source_files(root):
        if path.relative_to(root).parts[0] == "tests":
            continue  # Fixture paths are not runtime configuration references.
        try:
            tree = ast.parse(path.read_bytes())
        except (OSError, SyntaxError, UnicodeError, ValueError):
            continue
        relative = path.relative_to(root)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and node.value.startswith("configs/") and node.value.endswith((".yaml", ".yml"))):
                references.add((str(relative), node.lineno, node.value))
            if (relative == Path("scripts/train.py") and isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "CONFIGS" for target in node.targets)):
                try:
                    mapping = ast.literal_eval(node.value)
                    if not isinstance(mapping, dict) or not mapping or not all(
                        isinstance(value, str) for value in mapping.values()
                    ):
                        raise ValueError("expected a nonempty task-to-path mapping")
                    for value in mapping.values():
                        references.add((str(relative), node.lineno, "configs/" + value))
                except (ValueError, TypeError, SyntaxError) as exc:
                    errors.append(f"{relative}:{node.lineno}: cannot resolve CONFIGS: {exc}")
    for source, line, name in sorted(references):
        path = (root / name).resolve()
        if not path.is_relative_to(root / "configs") or not path.is_file():
            errors.append(f"{source}:{line}: missing local configuration {name}")
    return len(references), errors


def check_text_hygiene(root: Path, private_identifiers=()) -> tuple[int, list[str]]:
    private_handles = (re.compile(r"\b(?:" + "|".join(re.escape(item) for item in private_identifiers)
                                  + r")\b", re.IGNORECASE) if private_identifiers else None)
    errors = []
    count = 0
    for path in repository_files(root):
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue  # Never read entire weights/media into memory to detect text.
        try:
            text = path.read_bytes().decode("utf-8")
        except (UnicodeError, OSError):
            continue
        if "\0" in text:
            continue
        count += 1
        relative = path.relative_to(root)
        legal = (relative.parts[0] == "LICENSES" or path.name in
                 {"LICENSE", "NOTICE", "COPYING", "THIRD_PARTY_NOTICES.md", "ASSET_NOTES.md"})
        if not legal and private_handles and private_handles.search(str(relative)):
            errors.append(f"{relative}: private account identifier in filename")
        for number, line in enumerate(text.splitlines(), 1):
            if MACHINE_PATH.search(line):
                errors.append(f"{relative}:{number}: machine-specific path")
            if not legal and private_handles and private_handles.search(line):
                errors.append(f"{relative}:{number}: private account identifier")
        if CONFLICT_MARKER.search(text):
            errors.append(f"{relative}: unresolved merge conflict")
    return count, errors


def module_exists(root: Path, module: str) -> bool:
    path = root.joinpath(*module.split("."))
    return path.with_suffix(".py").is_file() or path.is_dir()


def check_local_imports(root: Path) -> list[str]:
    """Resolve explicit first-party imports and Gym entry points, without imports.

    External packages and imported attribute names are outside this static check.
    The data remappers also support bare sibling imports when run as scripts.
    """
    errors = []
    for path in source_files(root):
        try:
            tree = ast.parse(path.read_bytes())
        except (OSError, SyntaxError, UnicodeError, ValueError):
            continue  # Syntax errors are reported separately.
        relative = path.relative_to(root)
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parent = relative.parent.parts
                    prefix = parent[:len(parent) - node.level + 1]
                    modules = [".".join((*prefix, *node.module.split(".")))
                               if node.module else ".".join(prefix)]
                elif node.module:
                    modules = [node.module]
            elif isinstance(node, ast.Call):
                # Gym dynamically imports entry_point strings when making an env.
                modules = [keyword.value.value.split(":", 1)[0] for keyword in node.keywords
                           if keyword.arg == "entry_point" and isinstance(keyword.value, ast.Constant)
                           and isinstance(keyword.value.value, str)]
            for module in modules:
                first = module.split(".", 1)[0]
                is_local = first in LOCAL_ROOTS or first in REMOVED_ROOTS or first.startswith("inference_")
                if (is_local and not module_exists(root, module)
                        and not (path.parent / (module + ".py")).is_file()):
                    errors.append(f"{relative}:{node.lineno}: missing local module {module}")
    return sorted(set(errors))


def check_tracked_files(root: Path) -> tuple[int | None, list[str]]:
    """Check versioned files, ignoring local model/data downloads and .env files."""
    if not has_repository_index(root):
        return None, []
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None, []
    names = [os.fsdecode(name) for name in result.stdout.split(b"\0") if name]
    errors = []
    for name in names:
        path = Path(name)
        if path.suffix in FORBIDDEN_SUFFIXES or path.name == ".DS_Store":
            errors.append(f"Generated/model file is tracked: {name}")
        if path.name == ".env" or path.name.startswith(".env.") and path.name != ".env.example":
            errors.append(f"Local environment file is tracked: {name}")
        local_path = root / path
        if local_path.is_symlink() and not local_path.exists():
            errors.append(f"Broken symlink is tracked: {name}")
    return len(names), errors


def check_assets(root: Path) -> list[str]:
    errors = []
    for name in SOKOBAN_SPRITES:
        path = root / "gymnasium/envs/sokoban/surface" / (name + ".png")
        try:
            with path.open("rb") as stream:
                header = stream.read(33)
            # PNG signature, IHDR dimensions, bit depth, and RGB color type.
            if (header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR"
                    or header[16:26] != b"\x00\x00\x00\x10\x00\x00\x00\x10\x08\x02"):
                errors.append(f"Unexpected Sokoban sprite header: {path.relative_to(root)}")
        except OSError as exc:
            errors.append(f"Required Sokoban sprite: {path.relative_to(root)}: {exc}")
    pacman = root / "gymnasium/envs/pacman_2d/assets/greyblue9_pacman_python"
    actors = [f"sprite/pacman-{direction} {frame}.gif" for direction in "udlr" for frame in range(1, 9)]
    actors += [f"sprite/ghost {frame}.gif" for frame in (1, 2)]
    tiles = ["blank", "wall-x", "wall-nub", "wall-straight-horiz", "wall-straight-vert"]
    tiles += ["wall-t-" + side for side in ("bottom", "top", "right", "left")]
    tiles += ["wall-corner-" + corner for corner in ("ul", "ur", "ll", "lr")]
    tiles += ["wall-end-" + side for side in ("b", "t", "r", "l")]
    for name in actors + [f"tiles/{tile}.gif" for tile in tiles]:
        path = pacman / name
        try:
            with path.open("rb") as stream:
                signature = stream.read(6)
            if signature not in (b"GIF87a", b"GIF89a"):
                errors.append(f"Unexpected Pacman sprite header: {path.relative_to(root)}")
        except OSError as exc:
            errors.append(f"Required Pacman sprite: {path.relative_to(root)}: {exc}")
    require_replay = root / "scripts/data/sokoban_v8_state_replay_res256.py"
    if not require_replay.is_file():
        errors.append("Missing dynamic Sokoban renderer: scripts/data/sokoban_v8_state_replay_res256.py")
    return errors


def main() -> int:
    count, errors = check_python(ROOT)
    errors.extend(check_public_files(ROOT))
    license_count, license_errors = check_license_copies(ROOT)
    errors.extend(license_errors)
    config_count, config_errors = check_config_references(ROOT)
    errors.extend(config_errors)
    text_count, text_errors = check_text_hygiene(ROOT, tuple(filter(None, os.environ.get("AGIMA_PRIVATE_IDENTIFIERS", "").split(","))))
    errors.extend(text_errors)
    errors.extend(check_local_imports(ROOT))
    errors.extend(check_assets(ROOT))
    document_count, document_errors = check_markdown_links(ROOT)
    errors.extend(document_errors)
    command_count, command_errors = check_readme_commands(ROOT)
    errors.extend(command_errors)
    tracked_count, tracked_errors = check_tracked_files(ROOT)
    errors.extend(tracked_errors)
    print(f"Parsed {count} Python source files without importing them.")
    print(f"Checked {license_count} license copies and {config_count} Python configuration references.")
    print(f"Checked text/path hygiene in {text_count} retained text files.")
    print(f"Checked local links in {document_count} Markdown files and {command_count} README Python commands.")
    if tracked_count is None:
        print("Tracked-file hygiene skipped: Git checkout unavailable.")
    else:
        print(f"Checked {tracked_count} tracked paths for generated files and broken links.")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Source, local-import, asset, and documentation checks passed. GPU behavior is not tested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
