"""
local_executor.py
=================
Robust shell execution harness for AI agents.

Designed after OpenCode's bash tool architecture (packages/core/src/tool/bash.ts):

  WHAT THIS PROVIDES
  ──────────────────
  • Single unified bash tool  — no separate Python executor needed.
    Python code runs via:  python3 -c "..."  or temp script files.

  • Risk-assessed permission system
      CRITICAL  → rm -rf, curl|sh, fork bomb, etc.
      HIGH      → sudo, dd, passwd, killall, force-push
      MODERATE  → rm, mv, git push, pip/npm install
      SAFE      → everything else
    Risk levels are displayed as supplementary context in permission prompts.
    Actual access control is driven by config-file rules (allow/ask/deny).
    Config rules take precedence: a "cat *: allow" rule will auto-approve
    "cat .env" even if the risk engine flags it as risky.
    Consecutive-denial detection: if the same exact command (or same base
    command) is denied twice, the 3rd attempt is auto-denied with a clear
    message. History is cleared after auto-deny so the 4th attempt prompts again.

  • Platform-aware shell selection
      Windows  →  Git Bash / MSYS2 / Cygwin (validated, with cmd.exe fallback)
      POSIX    →  SHELL  / /bin/sh
    Shell discovery is logged; each candidate is validated before use.
    Runtime fallback: if the primary shell fails to spawn, retries with cmd.exe.

  • Timeout with SIGTERM → SIGKILL escalation (3 s grace period)

  • Output capture with 1 MB safety cap + smart head/tail truncation
    Full output saved to temp file; agent can read it later if needed.
    Default: first 50 + last 250 lines (300 total) shown to LLM.

  • Combined stdout + stderr (combineOutput: true)

  • Working directory tracking across calls
    (cd /some/dir as a standalone command updates session cwd)

  • External path advisory warnings
    (absolute paths outside workspace flagged — advisory only)

  • Interactive command detection (vim, ssh, python REPL, etc.) → auto-deny

  • Clean structured output dict for LLM consumption
"""

from __future__ import annotations

import os
import re
import sys
import shlex
import time
import signal
import hashlib
import warnings
import json as _json
import logging
import tempfile
import platform
import subprocess
import threading
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Set, Tuple, Callable, Any, Literal, Dict, Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic.functional_validators import AfterValidator


# ══════════════════════════════════════════════════════════════════════════════
# § 1  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_TIMEOUT_MS: int = 2 * 60 * 1_000  # 2 min  — matches OpenCode default
MAX_TIMEOUT_MS: int = 10 * 60 * 1_000  # 10 min — hard ceiling
MAX_CAPTURE_BYTES: int = 1 * 1024 * 1024  # 1 MB — hard safety cap for output capture
MAX_OUTPUT_LINES: int = 300  # default max lines returned to LLM
HEAD_LINES: int = 50  # lines from start shown on success
TAIL_LINES: int = 250  # lines from end shown on success
TAIL_LINES_ERROR: int = 280  # lines from end shown on failure (errors at bottom)
HEAD_LINES_ERROR: int = 20  # lines from start shown on failure
FORCE_KILL_DELAY_S: float = 3.0  # SIGTERM → SIGKILL grace period
OUTPUT_MAX_AGE_HOURS: int = 1  # auto-cleanup files older than this
MAX_VISIBLE_CHARS: int = 100_000  # byte-level cap for LLM-facing output
HEAD_CHARS: int = 2_000  # chars from start shown on success
TAIL_CHARS: int = 8_000  # chars from end shown on success
TAIL_CHARS_ERROR: int = 9_000  # chars from end shown on failure
HEAD_CHARS_ERROR: int = 1_000  # chars from start shown on failure

IS_WINDOWS: bool = platform.system() == "Windows"

log = logging.getLogger("local_executor")

# ══════════════════════════════════════════════════════════════════════════════
# § 5b  SHELL DISCOVERY  (Windows-only, production-grade)
# ══════════════════════════════════════════════════════════════════════════════

# Paths checked in order — covers default installs of Git Bash, MSYS2, Cygwin.
# User-scope installs use expanduser so they resolve per-user.
_BASH_CANDIDATES: List[str] = [
    # Git for Windows — standard install locations
    r"C:\Program Files\Git\bin\bash.exe",
    r"C:\Program Files (x86)\Git\bin\bash.exe",
    os.path.expanduser(r"~\AppData\Local\Programs\Git\bin\bash.exe"),
    # MSYS2 — standard install
    r"C:\msys64\usr\bin\bash.exe",
    r"C:\tools\msys64\usr\bin\bash.exe",
    os.path.expanduser(r"~\AppData\Local\Programs\msys64\usr\bin\bash.exe"),
    # Cygwin
    r"C:\cygwin64\bin\bash.exe",
    r"C:\cygwin\bin\bash.exe",
]

# WSL bash on Windows — different runtime, must NOT be used as a general shell.
_WSL_BASH_PATHS: Set[str] = {
    os.path.normpath(r"C:\Windows\System32\bash.exe"),
    os.path.normpath(
        os.path.expanduser(r"~\AppData\Local\Microsoft\WindowsApps\bash.exe")
    ),
}


def _is_wsl_bash(path: str) -> bool:
    """True if `path` points to WSL bash (launches a Linux VM, not a local shell)."""
    try:
        return os.path.normpath(path) in _WSL_BASH_PATHS
    except OSError:
        return False


def _find_bash_on_path() -> Optional[str]:
    """
    Search PATH for bash, excluding WSL bash.
    Returns the first valid non-WSL match, or None.
    """
    import shutil

    for name in ("bash", "bash.exe"):
        found = shutil.which(name)
        if found and not _is_wsl_bash(found):
            log.debug("shell discovery: found %s on PATH at %s", name, found)
            return found
    return None


def _validate_shell(path: str) -> bool:
    """
    Quick smoke test: spawn `path --version` with a short timeout.
    Returns True if the shell responds; False if it hangs, crashes, or isn't a shell.
    """
    try:
        proc = subprocess.Popen(
            [path, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        _, _ = proc.communicate(timeout=5)
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return False


def _find_git_bash() -> Optional[str]:
    """
    Discover a usable bash on Windows, prioritising Git Bash.

    Search order:
      1. Hardcoded candidate paths (Git Bash, MSYS2, Cygwin)
      2. PATH lookup (excludes WSL bash)
      3. SHELL env var (if it points to a valid, existing bash)

    Each candidate is validated with `bash --version` before being accepted.
    Falls back to None if nothing usable is found.

    Returns the absolute path, or None.
    """
    if not IS_WINDOWS:
        return None

    # 1. Hardcoded candidates
    for candidate in _BASH_CANDIDATES:
        if os.path.isfile(candidate):
            if _validate_shell(candidate):
                log.info("shell discovery: validated candidate %s", candidate)
                return candidate
            log.warning(
                "shell discovery: candidate %s exists but failed validation", candidate
            )

    # 2. PATH search (WSL excluded)
    path_match = _find_bash_on_path()
    if path_match:
        if _validate_shell(path_match):
            log.info("shell discovery: validated PATH match %s", path_match)
            return path_match
        log.warning("shell discovery: PATH match %s failed validation", path_match)

    # 3. SHELL env var
    env_shell = os.environ.get("SHELL", "")
    if env_shell and os.path.isfile(env_shell):
        base = os.path.basename(env_shell).lower()
        if "bash" in base or "sh" in base:
            if _validate_shell(env_shell):
                log.info("shell discovery: validated SHELL override %s", env_shell)
                return env_shell
            log.warning(
                "shell discovery: SHELL override %s failed validation", env_shell
            )

    log.warning(
        "shell discovery: no usable bash found on Windows, will fall back to cmd.exe"
    )
    return None


def _shell_is_cmd(shell_path: str) -> bool:
    """True if the resolved shell is cmd.exe / cmd."""
    name = os.path.basename(shell_path).lower()
    return name in ("cmd.exe", "cmd", "command.com")


def _shell_argv(shell_path: str, command: str) -> List[str]:
    """
    Build the argv list for spawning a command through the given shell.

    cmd.exe  →  [cmd.exe, /c, command]
    bash/sh  →  [bash,    -c, command]
    """
    if _shell_is_cmd(shell_path):
        return [shell_path, "/c", command]
    return [shell_path, "-c", command]


def _default_shell() -> str:
    """
    Platform default shell — mirrors OpenCode's defaultShell().

    On Windows, discovers Git Bash / MSYS2 / Cygwin (validated) before
    falling back to cmd.exe.  Override with the SHELL environment variable.

    On POSIX, returns $SHELL or /bin/sh.
    """
    # Explicit override via environment variable
    env_shell = os.environ.get("SHELL")
    if env_shell and os.path.isfile(env_shell):
        return env_shell

    if IS_WINDOWS:
        bash = _find_git_bash()
        if bash:
            return bash
        fallback = os.environ.get("COMSPEC", "cmd.exe")
        log.info("shell discovery: using cmd.exe fallback at %s", fallback)
        return fallback

    return env_shell or "/bin/sh"


import fnmatch

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
PURPLE = "\033[35m"
CYAN = "\033[36m"
DIM = "\033[2m"
SEP = "─" * 66


# ── Wildcard matching + cross-platform path normalization ────────────────────


def wildcard_match(pattern: str, value: str) -> bool:
    """Match `value` against a shell-style `pattern` using fnmatch.

    Normalizes to lowercase for consistent cross-platform behavior
    (fnmatch.fnmatch is case-insensitive on Windows, case-sensitive on POSIX).
    """
    return fnmatch.fnmatch(value.lower(), pattern.lower())


def _normalize_external_path(p: str) -> str:
    """Normalize a path for cross-platform external directory matching.

    Converts platform-specific temp directories to a canonical /tmp/ form
    so that patterns like /tmp/* work on Windows, macOS, and Linux.
    Patterns (containing wildcards) and Unix-style /tmp/ paths are NOT
    resolved against the filesystem.
    """
    expanded = os.path.expanduser(os.path.expandvars(p))

    if "*" in expanded or "?" in expanded:
        return expanded.replace("\\", "/")

    if expanded.startswith("/tmp/") or expanded == "/tmp":
        return expanded

    # macOS exposes /tmp as a symlink to /private/tmp; normalize both forms.
    macos_private_tmp = "/private/tmp"
    if expanded == macos_private_tmp or expanded.startswith(macos_private_tmp + "/"):
        remainder = expanded[len(macos_private_tmp) :]
        return f"/tmp{remainder}" if remainder else "/tmp"

    resolved = Path(expanded).resolve()

    try:
        system_temp = Path(tempfile.gettempdir()).resolve()
        temp_prefix = str(system_temp) + os.sep
        if str(resolved) == str(system_temp) or str(resolved).startswith(temp_prefix):
            remainder = str(resolved)[len(str(system_temp)) :]
            remainder = remainder.replace("\\", "/").lstrip("/")
            return f"/tmp/{remainder}" if remainder else "/tmp"
    except OSError:
        pass

    return str(resolved).replace("\\", "/")


# ══════════════════════════════════════════════════════════════════════════════
# § 2  RISK ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════


class RiskLevel(Enum):
    SAFE = 0
    MODERATE = 1
    HIGH = 2
    CRITICAL = 3

    def label(self) -> str:
        return {
            RiskLevel.SAFE: f"{GREEN}✓  SAFE{RESET}",
            RiskLevel.MODERATE: f"{YELLOW}⚠  MODERATE RISK{RESET}",
            RiskLevel.HIGH: f"{RED}⛔  HIGH RISK{RESET}",
            RiskLevel.CRITICAL: f"{PURPLE}💀  CRITICAL RISK{RESET}",
        }[self]


# (regex_pattern, RiskLevel, human_readable_reason)
# Ordered so the highest-risk check that matches wins.
RISK_PATTERNS: List[Tuple[str, RiskLevel, str]] = [
    # ── CRITICAL ──────────────────────────────────────────────────────────────
    (
        r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f\b",
        RiskLevel.CRITICAL,
        "recursive force-delete (rm -rf)",
    ),
    (
        r"\brm\s+-[a-zA-Z]*f[a-zA-Z]*r\b",
        RiskLevel.CRITICAL,
        "recursive force-delete (rm -fr)",
    ),
    (
        r"\brm\b.*\s+-[a-zA-Z]*r\b.*\s+-[a-zA-Z]*f\b",
        RiskLevel.CRITICAL,
        "recursive force-delete (rm -r -f)",
    ),
    (
        r"\brm\b.*\s+-[a-zA-Z]*f\b.*\s+-[a-zA-Z]*r\b",
        RiskLevel.CRITICAL,
        "recursive force-delete (rm -f -r)",
    ),
    (
        r"curl\s+[^\|]+\|\s*(ba)?sh",
        RiskLevel.CRITICAL,
        "remote code execution via curl pipe",
    ),
    (
        r"wget\s+[^\|]+\|\s*(ba)?sh",
        RiskLevel.CRITICAL,
        "remote code execution via wget pipe",
    ),
    (r":\(\)\s*\{[^}]*:\s*\|", RiskLevel.CRITICAL, "fork bomb pattern"),
    (r"\bmkfs\b", RiskLevel.CRITICAL, "filesystem format (mkfs)"),
    (r">\s*/dev/[sh]d[a-z]", RiskLevel.CRITICAL, "direct disk write (>/dev/sdX)"),
    (r">\s*/etc/", RiskLevel.CRITICAL, "write to /etc"),
    (r"\bfdisk\b|\bparted\b|\bgdisk\b", RiskLevel.CRITICAL, "disk partition tool"),
    (r"\bshred\b", RiskLevel.CRITICAL, "secure file deletion (shred)"),
    (r"\bformat\s+[a-zA-Z]:", RiskLevel.CRITICAL, "Windows drive format"),
    (
        r"\bdd\b.*\bof=/dev/[sh]d[a-z]",
        RiskLevel.CRITICAL,
        "disk overwrite via dd (of=/dev/sdX)",
    ),
    # ── HIGH ──────────────────────────────────────────────────────────────────
    (r"\bsudo\b", RiskLevel.HIGH, "superuser execution (sudo)"),
    (r"\bsu\s+-\b|\bsu\s+root\b", RiskLevel.HIGH, "switch to root user"),
    (r"\bdd\s+if=", RiskLevel.HIGH, "disk copy (dd)"),
    (r"\bpasswd\b", RiskLevel.HIGH, "password modification"),
    (
        r"\buseradd\b|\busermod\b|\buserdel\b",
        RiskLevel.HIGH,
        "user account modification",
    ),
    (r"\bchown\s+root\b", RiskLevel.HIGH, "ownership change to root"),
    (r"\biptables\b|\bnftables\b|\bpf\b", RiskLevel.HIGH, "firewall rule modification"),
    (r"\bsystemctl\s+(stop|disable|mask)\b", RiskLevel.HIGH, "stopping system service"),
    (r"\bkillall\b|\bpkill\b", RiskLevel.HIGH, "bulk process termination"),
    (r"\beval\b", RiskLevel.HIGH, "dynamic code execution via eval"),
    (r"\bexec\b", RiskLevel.HIGH, "dynamic code execution via exec"),
    (r"\bnc\b|\bnetcat\b", RiskLevel.HIGH, "network listener (possible reverse shell)"),
    (r"\bcrontab\s+-r\b", RiskLevel.HIGH, "remove all cron jobs"),
    (
        r"\bnpm\s+publish\b|\bpip\s+upload\b",
        RiskLevel.HIGH,
        "package registry publishing",
    ),
    (r"\bgit\s+push\s+[^\n]*--force\b", RiskLevel.HIGH, "force git push (data loss)"),
    (r"\bchattr\b", RiskLevel.HIGH, "file attribute change (chattr)"),
    # ── MODERATE ──────────────────────────────────────────────────────────────
    (r"\brm\s+", RiskLevel.MODERATE, "file deletion (rm)"),
    (r"\bmv\s+", RiskLevel.MODERATE, "file move/rename (mv)"),
    (r"\btruncate\b", RiskLevel.MODERATE, "file truncation"),
    (
        r"\bchmod\s+[0-7]*7[0-7][0-7]\b",
        RiskLevel.MODERATE,
        "world-writable permissions",
    ),
    (
        r"\bgit\s+reset\s+--hard\b",
        RiskLevel.MODERATE,
        "hard git reset (discards changes)",
    ),
    (r"\bgit\s+clean\s+-[a-zA-Z]*f\b", RiskLevel.MODERATE, "force git clean"),
    (r"\bgit\s+push\b", RiskLevel.MODERATE, "git push"),
    (
        r"\bgit\s+(?!(status|log|diff|branch|show)\b)\w+",
        RiskLevel.MODERATE,
        "git command that may modify repository state",
    ),
    (r"\bnpm\s+install\b|\bnpm\s+i\b", RiskLevel.MODERATE, "npm install"),
    (r"\bpip\s+install\b|\bpip3\s+install\b", RiskLevel.MODERATE, "pip install"),
    (
        r"\bapt(-get)?\s+(install|remove|purge)\b",
        RiskLevel.MODERATE,
        "apt package manager",
    ),
    (r"\bbrew\s+install\b", RiskLevel.MODERATE, "Homebrew install"),
    (r"\byarn\s+add\b", RiskLevel.MODERATE, "yarn add"),
    (r"\bhistory\s+-[cw]\b", RiskLevel.MODERATE, "clear shell history"),
]

# Commands that block waiting for stdin — they'd hang until timeout.
INTERACTIVE_PATTERNS: List[Tuple[str, str]] = [
    (r"^python3?\s+-i\b", "interactive Python REPL (-i flag)"),
    (r"^python3?\s*$", "interactive Python REPL"),
    (r"^(i?python3?|bpython)\s*$", "interactive Python REPL"),
    (r"^(ba|z|fi|da|k)?sh\s*$", "interactive shell"),
    (r"^ssh\s", "SSH session (interactive)"),
    (r"\bvim?\b|\bnano\b|\bemacs\b", "terminal text editor"),
    (r"\bless\b|\bmore\b|\bman\b", "terminal pager"),
    (r"\btop\b|\bhtop\b|\bbtop\b|\bglances\b", "interactive system monitor"),
    (r"\bmysql\b|\bpsql\b|\bsqlite3\b", "interactive database REPL"),
    (r"^node\s+--interactive\b", "interactive Node.js REPL (--interactive flag)"),
    (r"\bnode\s*$|\bts-node\s*$", "interactive Node.js REPL"),
    (r"^ruby\s*$", "interactive Ruby REPL"),
    (r"\birb\s*$|\bpry\s*$", "interactive Ruby REPL"),
]


def assess_risk(command: str) -> Tuple[RiskLevel, List[str]]:
    """
    Scan command string for risky patterns.
    Returns (highest_risk_level, [list_of_reasons]).
    """
    highest = RiskLevel.SAFE
    reasons: List[str] = []

    for pattern, level, description in RISK_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            if level.value > highest.value:
                highest = level
            if description not in reasons:
                reasons.append(description)

    return highest, reasons


def is_interactive(command: str) -> Tuple[bool, str]:
    """Returns (True, reason) if command would likely block for terminal input."""
    stripped = command.strip()
    for pattern, reason in INTERACTIVE_PATTERNS:
        if re.search(pattern, stripped, re.IGNORECASE):
            return True, reason
    return False, ""


# ══════════════════════════════════════════════════════════════════════════════
# § 2b  OUTPUT HANDLING  (smart truncation + temp file preservation)
# ══════════════════════════════════════════════════════════════════════════════


_output_dir: Optional[Path] = None
_output_dir_lock = threading.Lock()


def _get_output_dir() -> Path:
    """Get or create a secure temp output directory for saving full command output.

    Uses tempfile.mkdtemp() for a unique, unpredictable directory name.
    On POSIX, sets permissions to 0700 (owner-only access) to prevent
    other users from reading command output that may contain secrets.
    """
    global _output_dir
    with _output_dir_lock:
        if _output_dir is not None and _output_dir.is_dir():
            return _output_dir
        base = Path(tempfile.gettempdir())
        d = Path(tempfile.mkdtemp(prefix="local_executor_", dir=base))
        if not IS_WINDOWS:
            try:
                import stat

                d.chmod(stat.S_IRWXU)  # owner only: rwx------
            except OSError:
                pass
        _output_dir = d
        return d


def _save_output_to_file(output: str, command: str) -> Optional[str]:
    """
    Save captured command output to a temp file.

    Returns the file path string, or None on error.
    The agent can later read this file if it needs the preserved captured output.
    On POSIX, files are created with 0600 permissions (owner read/write only)
    to prevent other users from reading command output that may contain secrets.
    """
    try:
        cmd_hash = hashlib.sha1(command.encode()).hexdigest()[:6]
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{cmd_hash}.log"
        path = _get_output_dir() / filename
        path.write_text(output, encoding="utf-8", errors="replace")
        if not IS_WINDOWS:
            try:
                import stat

                path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # owner read/write only
            except OSError:
                pass
        return str(path)
    except OSError:
        return None


def _smart_truncate(
    output: str, exit_code: Optional[int]
) -> Tuple[str, bool, bool, int, int]:
    """
    Apply head+tail truncation to output.

    Returns (truncated_text, line_truncated, char_truncated, total_lines, total_chars).
    - Line truncation: when total_lines > MAX_OUTPUT_LINES
    - Char truncation: when total_chars > MAX_VISIBLE_CHARS (safety net for long lines/gigantic lines)
    """
    total_chars = len(output)
    total_lines = output.count("\n") + 1

    if total_chars <= MAX_VISIBLE_CHARS and total_lines <= MAX_OUTPUT_LINES:
        return output, False, False, total_lines, total_chars

    if exit_code is not None and exit_code != 0:
        head_n, tail_n = HEAD_LINES_ERROR, TAIL_LINES_ERROR
        head_c, tail_c = HEAD_CHARS_ERROR, TAIL_CHARS_ERROR
    else:
        head_n, tail_n = HEAD_LINES, TAIL_LINES
        head_c, tail_c = HEAD_CHARS, TAIL_CHARS

    line_truncated = False
    char_truncated = False

    # Step 1: Line cap (produces cleaner output when there are many small lines)
    if total_lines > MAX_OUTPUT_LINES:
        lines = output.splitlines(keepends=True)
        head = lines[:head_n]
        tail = lines[-tail_n:]
        omitted = total_lines - head_n - tail_n
        truncated = "".join(head)
        truncated += f"\n... ({omitted} lines omitted) ...\n\n"
        truncated += "".join(tail)
        line_truncated = True
    else:
        truncated = output

    # Step 2: Char cap — catches the case where a single line is 512 KB+
    # (line truncation can't protect against this since it operates on whole lines)
    if len(truncated) > MAX_VISIBLE_CHARS:
        head = truncated[:head_c]
        tail = truncated[-tail_c:]
        omitted_chars = len(truncated) - head_c - tail_c
        truncated = head
        truncated += f"\n... ({omitted_chars} chars omitted) ...\n\n"
        truncated += tail
        char_truncated = True

    return truncated, line_truncated, char_truncated, total_lines, total_chars


def _cleanup_old_outputs() -> None:
    """Delete output files older than OUTPUT_MAX_AGE_HOURS."""
    try:
        cutoff = time.time() - (OUTPUT_MAX_AGE_HOURS * 3600)
        for f in _get_output_dir().iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
    except OSError:
        pass


def _line_window(exit_code: Optional[int]) -> Tuple[int, int]:
    """Return the head/tail line window used for smart truncation formatting."""
    if exit_code is not None and exit_code != 0:
        return HEAD_LINES_ERROR, TAIL_LINES_ERROR
    return HEAD_LINES, TAIL_LINES


def _char_window(exit_code: Optional[int]) -> Tuple[int, int]:
    """Return the head/tail char window used for smart truncation formatting."""
    if exit_code is not None and exit_code != 0:
        return HEAD_CHARS_ERROR, TAIL_CHARS_ERROR
    return HEAD_CHARS, TAIL_CHARS


def _format_truncation_suffix(
    was_truncated: bool,
    total_lines: int,
    exit_code: Optional[int],
    output_file: Optional[str],
    stderr_output_file: Optional[str] = None,
    char_truncated: bool = False,
    total_chars: int = 0,
    line_truncated: bool = False,
) -> str:
    """Format the metadata suffix appended to truncated output blocks."""
    suffix = ""
    if was_truncated:
        if line_truncated:
            head_n, tail_n = _line_window(exit_code)
            suffix += (
                f"\n[truncated: {total_lines} total lines, "
                f"showing first {head_n} + last {tail_n}]"
            )
        if char_truncated:
            head_c, tail_c = _char_window(exit_code)
            suffix += (
                f"\n[truncated: {total_chars} total chars, "
                f"showing first {head_c} + last {tail_c} chars]"
            )
    if output_file:
        suffix += f"\n[full output saved to: {output_file}]"
    if stderr_output_file:
        suffix += f"\n[full stderr saved to: {stderr_output_file}]"
    return suffix


def _compose_visible_output(
    output: str,
    stdout: Optional[str],
    stderr: Optional[str],
) -> str:
    """Build the text block shown to the model from merged or split streams."""
    if stdout is not None or stderr is not None:
        parts: List[str] = []
        if stdout:
            parts.append(f"[stdout]\n{stdout}")
        if stderr:
            parts.append(f"[stderr]\n{stderr}")
        return "\n\n".join(parts)
    return output or ""


# ══════════════════════════════════════════════════════════════════════════════
# § 3  PYDANTIC MODELS  (matching OpenCode's bash.ts schema architecture)
# ══════════════════════════════════════════════════════════════════════════════

# ── Outcome enum (mirrors ADK's code_execution_result) ──────────────────────


class Outcome(str, Enum):
    """Execution outcome for LLM consumption."""

    OK = "OUTCOME_OK"
    ERROR = "OUTCOME_ERROR"
    TIMEOUT = "OUTCOME_TIMEOUT"
    DENIED = "OUTCOME_DENIED"


# ── Input schema (mirrors OpenCode's Input = Schema.Struct({...})) ──────────


def _non_empty_strip(v: str) -> str:
    """AfterValidator: reject empty or whitespace-only strings, strip trailing whitespace."""
    v = v.rstrip()
    if not v:
        raise ValueError("command must not be empty or whitespace-only")
    return v


MAX_COMMAND_LENGTH: int = 100_000  # 100 KB — generous but prevents multi-MB abuse


def _validate_command_length(v: str) -> str:
    """AfterValidator: reject commands exceeding MAX_COMMAND_LENGTH."""
    if len(v) > MAX_COMMAND_LENGTH:
        raise ValueError(
            f"Command too long ({len(v)} chars, max {MAX_COMMAND_LENGTH}). "
            "Break into smaller commands or write to a script file."
        )
    return v


class BashInput(BaseModel):
    """
    Input schema for the bash tool.

    Matches OpenCode's Input:
      command : string  — shell command to execute
      workdir : string? — working directory (defaults to workspace root)
      timeout : int?    — timeout in ms (default 120000, max 600000)
    """

    command: Annotated[
        str,
        AfterValidator(_non_empty_strip),
        AfterValidator(_validate_command_length),
    ] = Field(
        ...,
        description="Shell command string to execute. Supports full shell syntax: "
        "pipes, redirects, &&, ||, subshells, etc.",
    )
    workdir: Optional[str] = Field(
        default=None,
        description="Working directory. Defaults to the active workspace. "
        "Relative paths resolve from the workspace root.",
    )
    timeout: Optional[int] = Field(
        default=None,
        ge=1,
        le=MAX_TIMEOUT_MS,
        description=f"Timeout in milliseconds. Default: {DEFAULT_TIMEOUT_MS}, "
        f"maximum: {MAX_TIMEOUT_MS}.",
    )


# ── StructuredOutput (compact metadata — what the model sees) ───────────────


class BashStructuredOutput(BaseModel):
    """
    Compact metadata returned alongside the output text.

    Matches OpenCode's StructuredOutput:
      exit     : number?  — exit code (undefined on timeout)
      truncated: boolean  — was output truncated at 1 MB?
      timeout  : boolean? — did it time out?

    TODO: Wire this into BashOutput as the compact metadata block instead of
    embedding fields directly. Currently unused outside type declarations.
    """

    exit: Optional[int] = Field(
        default=None,
        description="Process exit code. None if the command timed out.",
    )
    truncated: bool = Field(
        default=False,
        description="True if output was truncated at the 1 MB safety limit.",
    )
    timeout: Optional[bool] = Field(
        default=None,
        description="True if the command exceeded the timeout and was killed.",
    )


# ── Output (full result — extends StructuredOutput) ─────────────────────────


class BashOutput(BaseModel):
    """
    Full output returned by the bash tool.

    Matches OpenCode's Output = Schema.Struct({
      ...StructuredOutput.fields,
      output:  Schema.String,
      warnings: Schema.Array(Schema.String).pipe(Schema.optional),
    })

    The to_model_output() method converts this to the LLM-facing format:
      [ { type: "text", text: output.output },
        { type: "text", text: "Command exited with code 0." } ]
    """

    # StructuredOutput fields
    exit: Optional[int] = Field(default=None)
    truncated: bool = Field(default=False)
    timeout: Optional[bool] = Field(default=None)
    # Extended fields
    output: str = Field(
        default="",
        description="Combined stdout + stderr of the command.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Advisory warnings about external directory references.",
    )
    output_file: Optional[str] = Field(
        default=None,
        description="Path to temp file containing the preserved captured output.",
    )
    stderr_output_file: Optional[str] = Field(
        default=None,
        description="Path to temp file containing the preserved captured stderr (combine_output=False).",
    )
    stdout: Optional[str] = Field(
        default=None,
        description="Stdout only (when combine_output=False). None if streams are merged.",
    )
    stderr: Optional[str] = Field(
        default=None,
        description="Stderr only (when combine_output=False). None if streams are merged.",
    )
    stderr_truncated: bool = Field(
        default=False,
        description="True if stderr exceeded the 1 MB safety limit (when combine_output=False).",
    )
    denied: bool = Field(
        default=False,
        description="True if the command was denied by the permission system.",
    )

    def to_model_output(self) -> List[dict]:
        """
        Convert to OpenCode's toModelOutput format.

        TODO: Switch toolkit to use this method (matches OpenCode's exact format)
        instead of to_adk_dict(). Currently unused — the toolkit uses to_adk_dict().

        Returns a list of text blocks:
          1. Raw command output (smart-truncated: head + tail)
          2. Summary line: "Command exited with code X." or "Command timed out."

        If output was truncated, a note with the file path is included so
        the agent can read the full output later if needed.
        """
        parts: List[dict] = []

        # Block 1: warnings (if any)
        if self.warnings:
            warnings_text = "Warnings:\n" + "\n".join(f"- {w}" for w in self.warnings)
            parts.append({"type": "text", "text": warnings_text})

        # Block 2: smart-truncated output
        visible_output = _compose_visible_output(self.output, self.stdout, self.stderr)
        truncated_text, line_truncated, char_truncated, total_lines, total_chars = (
            _smart_truncate(visible_output or "(no output)", self.exit)
        )
        was_truncated = line_truncated or char_truncated or self.truncated
        truncated_text += _format_truncation_suffix(
            was_truncated,
            total_lines,
            self.exit,
            self.output_file,
            stderr_output_file=self.stderr_output_file,
            char_truncated=char_truncated,
            total_chars=total_chars,
            line_truncated=line_truncated,
        )

        parts.append({"type": "text", "text": truncated_text})

        # Block 3: summary line
        summary = self._summary_line()
        parts.append({"type": "text", "text": summary})

        return parts

    def _summary_line(self) -> str:
        """Generate the summary line matching OpenCode's modelOutput function."""
        if self.timeout:
            return "Command timed out before completion."
        if self.exit is not None:
            return f"Command exited with code {self.exit}."
        return "Command failed before producing an exit code."

    def to_adk_dict(self) -> dict:
        """
        ADK-compatible dict format.

        Returns the structured fields as a flat dict for agent frameworks
        that expect a single dict response. Output is smart-truncated
        (head+tail) to keep the LLM context manageable.
        """
        visible_output = _compose_visible_output(self.output, self.stdout, self.stderr)
        truncated_output, line_truncated, char_truncated, total_lines, total_chars = (
            _smart_truncate(visible_output or "(no output)", self.exit)
        )
        capture_truncated = self.truncated
        was_truncated = line_truncated or char_truncated or self.truncated
        truncated_output += _format_truncation_suffix(
            was_truncated,
            total_lines,
            self.exit,
            self.output_file,
            stderr_output_file=self.stderr_output_file,
            char_truncated=char_truncated,
            total_chars=total_chars,
            line_truncated=line_truncated,
        )

        # When combine_output=False, the composed output already contains
        # [stdout]/[stderr] labeled blocks. Don't duplicate stdout/stderr
        # as separate dict fields — the ADK serializes ALL dict keys into
        # the LLM conversation, and 3x duplication (output + stdout + stderr)
        # causes rapid context overflow. The composed output is the authority.
        if self.stdout is not None or self.stderr is not None:
            # Separate-streams mode: output already has the composed version
            dict_stdout = None
            dict_stderr = None
        else:
            # Merged mode: no separate streams, so nothing to exclude
            dict_stdout = None
            dict_stderr = None

        return {
            "outcome": self._outcome().value,
            "output": truncated_output,
            "exit_code": self.exit,
            "success": self.exit is not None and self.exit == 0,
            "denied": self.denied,
            "truncated": was_truncated,
            "capture_truncated": capture_truncated,
            "line_truncated": line_truncated,
            "char_truncated": char_truncated,
            "timed_out": self.timeout or False,
            "warnings": self.warnings,
            "summary": self._summary_line(),
            "output_file": self.output_file,
            "stderr_output_file": self.stderr_output_file,
            "stdout": dict_stdout,
            "stderr": dict_stderr,
            "stderr_truncated": self.stderr_truncated,
        }

    def _outcome(self) -> Outcome:
        """Derive the Outcome enum from the structured fields."""
        if self.timeout:
            return Outcome.TIMEOUT
        if self.exit is None:
            return Outcome.ERROR
        if self.exit == 0:
            return Outcome.OK
        return Outcome.ERROR


# ── Permission models ───────────────────────────────────────────────────────


class PermissionDecision(Enum):
    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    DENY = "deny"


@dataclass
class _PendingPermissionPrompt:
    """Tracks an in-flight permission prompt so concurrent callers can share it."""

    event: threading.Event
    decision: Optional[PermissionDecision] = None


class PermissionConfig(BaseModel):
    """
    Deprecated permission policy configuration.

    DEPRECATED: Use LocalExecutorConfig instead.

    Only auto_approve is read by _convert_permission_config().
    The following fields are DEFINED but SILENTLY IGNORED:
      min_ask_level — has no effect (risk level does not gate access)
      deny_critical — has no effect (config rules are the authority)
      allowed_actions, denied_actions — has no effect (unused)
    """

    auto_approve: bool = Field(
        default=False,
        description="Skip all permission prompts. Use for testing/CI.",
    )
    min_ask_level: RiskLevel = Field(
        default=RiskLevel.MODERATE,
        description="Commands below this risk level are auto-allowed.",
    )
    deny_critical: bool = Field(
        default=False,
        description="If True, CRITICAL commands are auto-denied without asking.",
    )
    allowed_actions: List[str] = Field(
        default_factory=list,
        description="Actions always allowed without prompting (e.g. ['python_executor']).",
    )
    denied_actions: List[str] = Field(
        default_factory=list,
        description="Actions always denied without prompting (e.g. ['bash']).",
    )


# ── CommandResult (internal, used by ProcessRunner) ─────────────────────────


@dataclass
class CommandResult:
    """
    Internal result from ProcessRunner.run().

    This is the raw execution result before formatting for the LLM.
    The BashOutput model is constructed from this.
    """

    command: str
    exit_code: Optional[int]  # None on timeout
    output: str  # combined stdout + stderr (UTF-8), or stdout only when combine_output=False
    truncated: bool  # output hit the 1 MB cap
    timed_out: bool
    cwd: str  # the directory the command ran in
    warnings: List[str] = field(default_factory=list)
    denied: bool = False
    output_file: Optional[str] = None
    stderr_output_file: Optional[str] = None  # stderr file path (combine_output=False)
    stdout: Optional[str] = None  # stdout only (when combine_output=False)
    stderr: Optional[str] = None  # stderr only (when combine_output=False)
    stderr_truncated: bool = (
        False  # stderr hit the 1 MB cap (when combine_output=False)
    )

    def to_bash_output(self) -> BashOutput:
        """Convert to the Pydantic Output model for LLM consumption."""
        return BashOutput(
            exit=self.exit_code if not self.timed_out else None,
            truncated=self.truncated,
            timeout=self.timed_out if self.timed_out else None,
            output=self.output if self.stdout is None else "",
            warnings=self.warnings,
            output_file=self.output_file,
            stderr_output_file=self.stderr_output_file,
            stdout=self.stdout,
            stderr=self.stderr,
            stderr_truncated=self.stderr_truncated,
            denied=self.denied,
        )

    def to_dict(self) -> dict:
        """Backwards-compatible dict output."""
        return self.to_bash_output().to_adk_dict()


# ══════════════════════════════════════════════════════════════════════════════
# § 4  PERMISSION MANAGER
# ══════════════════════════════════════════════════════════════════════════════

# Commands that have subcommands — base extraction uses first two words.
# e.g. "git rm file" → "git rm",  "pip install numpy" → "pip install"
_PREFIX_COMMANDS: frozenset = frozenset(
    {
        "git",
        "sudo",
        "npm",
        "npx",
        "pip",
        "pip3",
        "yarn",
        "apt",
        "apt-get",
        "brew",
        "systemctl",
        "docker",
    }
)


def _extract_command_base(command: str) -> str:
    """
    Extract the base command name from a command string.

    Used for session-level allow (option 3) and consecutive-denial tracking.

    Examples:
        "rm -rf artifacts"       → "rm"
        "rm"                     → "rm"
        "git rm README.md"       → "git rm"
        "git push origin main"   → "git push"
        "pip install numpy"      → "pip install"
        "sudo rm -rf /"          → "sudo rm"
        "ls -la"                 → "ls"
    """
    tokens = command.split()
    if not tokens:
        return command
    cmd = tokens[0].lower()
    if cmd in _PREFIX_COMMANDS and len(tokens) > 1:
        return f"{tokens[0]} {tokens[1]}"
    return tokens[0]


# ── New config-file-based permission models ─────────────────────────────────

PermissionAction = Literal["allow", "ask", "deny"]


class BashPermissionRules(BaseModel):
    """Ordered list of (pattern, action) pairs for bash commands."""

    model_config = ConfigDict(frozen=True)
    rules: List[Tuple[str, PermissionAction]] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, PermissionAction]) -> "BashPermissionRules":
        return cls(rules=list(d.items()))

    def evaluate(self, command: str) -> PermissionAction:
        result: PermissionAction = "ask"
        for pattern, action in self.rules:
            if wildcard_match(pattern, command):
                result = action
        return result


class ExternalDirectoryRules(BaseModel):
    """Permission rules for external directory access."""

    model_config = ConfigDict(frozen=True)
    rules: List[Tuple[str, PermissionAction]] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, PermissionAction]) -> "ExternalDirectoryRules":
        return cls(rules=list(d.items()))

    def evaluate(self, directory: str) -> PermissionAction:
        result: PermissionAction = "ask"
        canonical = _normalize_external_path(directory)
        for raw_pattern, action in self.rules:
            pattern = _normalize_external_path(raw_pattern)
            if wildcard_match(pattern, canonical):
                result = action
        return result


class PermissionBlock(BaseModel):
    """The full permission configuration block."""

    model_config = ConfigDict(frozen=True)
    global_default: PermissionAction = "ask"
    bash: BashPermissionRules = Field(default_factory=BashPermissionRules)
    external_directory: ExternalDirectoryRules = Field(
        default_factory=ExternalDirectoryRules
    )


class LocalExecutorConfig(BaseModel):
    """Full config loaded from local_executor.json files."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    shell: Optional[str] = None
    timeout_ms: Annotated[int, Field(ge=1, le=MAX_TIMEOUT_MS)] = 120_000
    permission: PermissionBlock = Field(default_factory=PermissionBlock)
    consecutive_deny_threshold: Annotated[int, Field(ge=1, le=10)] = 2
    non_interactive: bool = False

    @field_validator("shell", mode="before")
    @classmethod
    def _validate_shell(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        import shutil

        if not os.path.isabs(v):
            resolved = shutil.which(v)
            if resolved:
                return resolved
            raise ValueError(f"Shell '{v}' not found on PATH")
        if not os.path.isfile(v):
            raise ValueError(f"Shell path does not exist: {v}")
        return v


class ConfigLoader:
    """Loads and merges local_executor.json from multiple locations.

    Resolution order:
      1. Built-in defaults
      2. User config:    ~/.config/local_executor/local_executor.json
      3. Project config: {workspace_dir}/.local_executor/local_executor.json
    """

    USER_CONFIG_PATH: Path = (
        Path.home() / ".config" / "local_executor" / "local_executor.json"
    )

    def __init__(self, workspace_dir: str) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()

    def project_config_path(self) -> Path:
        return self.workspace_dir / ".local_executor" / "local_executor.json"

    def load(self) -> LocalExecutorConfig:
        """Load and deep-merge all config layers. Returns a frozen config object."""
        merged: Dict[str, Any] = self._builtin_defaults()

        for path in [self.USER_CONFIG_PATH, self.project_config_path()]:
            if path.exists():
                try:
                    data = self._read_json(path)
                    merged = self._deep_merge(merged, data)
                    log.info("Loaded config from %s", path)
                except Exception as exc:
                    log.warning("Failed to load config from %s: %s", path, exc)

        merged = self._apply_env_overrides(merged)

        if not self.project_config_path().exists():
            try:
                self._write_default_config()
            except Exception:
                pass

        return self._build_config(self._normalize_permission_block(merged))

    def _builtin_defaults(self) -> Dict[str, Any]:
        return {
            "timeout_ms": 120_000,
            "consecutive_deny_threshold": 2,
            "non_interactive": False,
            "permission": {
                "*": "ask",
                "bash": {
                    "*": "ask",
                    "ls": "allow",
                    "ls *": "allow",
                    "cat *": "allow",
                    "echo *": "allow",
                    "pwd": "allow",
                    "whoami": "allow",
                    "date": "allow",
                    "grep *": "allow",
                    "find *": "allow",
                    "which *": "allow",
                    "python3 -c *": "allow",
                    "git status*": "allow",
                    "git log*": "allow",
                    "git diff*": "allow",
                    "git branch*": "allow",
                },
                "external_directory": {
                    "*": "ask",
                    "/tmp/*": "allow",
                },
            },
        }

    def _write_default_config(self) -> None:
        config_path = self.project_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        default_content = {
            "$schema": "./local_executor.schema.json",
            "shell": None,
            "timeout_ms": 120000,
            "permission": {
                "*": "ask",
                "bash": {
                    "*": "ask",
                    "ls": "allow",
                    "ls *": "allow",
                    "cat *": "allow",
                    "echo *": "allow",
                    "pwd": "allow",
                    "whoami": "allow",
                    "date": "allow",
                    "grep *": "allow",
                    "find *": "allow",
                    "which *": "allow",
                    "python3 -c *": "allow",
                    "git status*": "allow",
                    "git log*": "allow",
                    "git diff*": "allow",
                    "git branch*": "allow",
                    "git push*": "deny",
                    "rm *": "ask",
                    "sudo *": "deny",
                },
                "external_directory": {
                    "*": "ask",
                    "/tmp/*": "allow",
                },
            },
            "consecutive_deny_threshold": 2,
            "non_interactive": False,
        }
        config_path.write_text(_json.dumps(default_content, indent=2), encoding="utf-8")

    def _read_json(self, path: Path) -> Dict[str, Any]:
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"(?m)^\s*//.*$", "", text)
        return _json.loads(text)

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        result = dict(base)
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = self._deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    def _apply_env_overrides(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply LE_* environment variable overrides.

        Supported env vars:
          LE_SHELL               → shell
          LE_TIMEOUT_MS          → timeout_ms (int)
          LE_NON_INTERACTIVE     → non_interactive (true/false/1/0)
          LE_PERMISSION_BASH     → JSON string for permission.bash dict
          LE_PERMISSION_DEFAULT  → global permission default (allow/ask/deny)
        """
        result = dict(data)
        if shell := os.environ.get("LE_SHELL"):
            result["shell"] = shell
        if timeout := os.environ.get("LE_TIMEOUT_MS"):
            try:
                result["timeout_ms"] = int(timeout)
            except ValueError:
                log.warning("Invalid LE_TIMEOUT_MS value: %s", timeout)
        if ni := os.environ.get("LE_NON_INTERACTIVE"):
            result["non_interactive"] = ni.lower() in ("true", "1", "yes")
        if bash_rules := os.environ.get("LE_PERMISSION_BASH"):
            try:
                perm = result.setdefault("permission", {})
                perm["bash"] = _json.loads(bash_rules)
            except _json.JSONDecodeError:
                log.warning("Invalid LE_PERMISSION_BASH JSON: %s", bash_rules[:100])
        if default_perm := os.environ.get("LE_PERMISSION_DEFAULT"):
            perm = result.setdefault("permission", {})
            perm["*"] = default_perm
        return result

    def _normalize_permission_block(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert raw dict into PermissionBlock-compatible structure."""
        raw_perm = data.get("permission", {})
        if isinstance(raw_perm, str):
            data["permission"] = PermissionBlock(
                global_default=raw_perm,  # type: ignore
                bash=BashPermissionRules(),
                external_directory=ExternalDirectoryRules(),
            )
            return data

        global_default = raw_perm.get("*", "ask")
        bash_dict = raw_perm.get("bash", {})
        ext_dict = raw_perm.get("external_directory", {})

        data["permission"] = PermissionBlock(
            global_default=global_default,  # type: ignore
            bash=BashPermissionRules.from_dict(bash_dict)
            if isinstance(bash_dict, dict)
            else BashPermissionRules(),
            external_directory=ExternalDirectoryRules.from_dict(ext_dict)
            if isinstance(ext_dict, dict)
            else ExternalDirectoryRules(),
        )
        return data

    def _build_config(self, data: Dict[str, Any]) -> LocalExecutorConfig:
        raw_perm = data.get("permission", {})

        # If permission is already a PermissionBlock (from _normalize_permission_block), use it directly
        if isinstance(raw_perm, PermissionBlock):
            permission = raw_perm
        else:
            global_default: PermissionAction = raw_perm.get("*", "ask")  # type: ignore
            bash_dict = raw_perm.get("bash", {})
            ext_dict = raw_perm.get("external_directory", {})
            permission = PermissionBlock(
                global_default=global_default,  # type: ignore
                bash=BashPermissionRules.from_dict(bash_dict)
                if isinstance(bash_dict, dict)
                else BashPermissionRules(),
                external_directory=ExternalDirectoryRules.from_dict(ext_dict)
                if isinstance(ext_dict, dict)
                else ExternalDirectoryRules(),
            )

        return LocalExecutorConfig(
            shell=data.get("shell"),
            timeout_ms=data.get("timeout_ms", 120_000),
            consecutive_deny_threshold=data.get("consecutive_deny_threshold", 2),
            non_interactive=data.get("non_interactive", False),
            permission=permission,
        )


class PermissionManager:
    """
    Three-outcome permission gate, mirroring OpenCode's PermissionV2:

        ALLOW_ONCE    — user said yes just for this invocation
        ALLOW_SESSION — user said yes; rule saved for the entire session
        DENY          — user said no; execution blocked

    Config-driven: uses LocalExecutorConfig rules (last-match-wins via fnmatch).
    Risk level (RiskLevel) is shown as supplementary info in the prompt but does
    not independently gate access — config rules are the authority.
    """

    def __init__(
        self,
        config: Optional[LocalExecutorConfig] = None,
        compat: Optional[PermissionConfig] = None,
    ):
        if compat is not None:
            warnings.warn(
                "PermissionConfig is deprecated; use LocalExecutorConfig instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if config is not None and compat is not None:
            log.warning(
                "PermissionConfig was provided alongside LocalExecutorConfig and will be ignored."
            )
        if config is not None:
            self._config = config
        elif compat is not None:
            self._config = self._convert_permission_config(compat)
        else:
            self._config = LocalExecutorConfig()
        self._compat = compat
        self._session_allows: Dict[str, PermissionAction] = {}
        self._session_ext_allows: Set[str] = set()
        self._denial_history: List[Tuple[str, str]] = []
        self._pending_prompts: Dict[str, _PendingPermissionPrompt] = {}
        self._lock = threading.Lock()
        self._last_denial_message: Optional[str] = None

    @staticmethod
    def _convert_permission_config(pc: PermissionConfig) -> LocalExecutorConfig:
        """Convert old-style PermissionConfig to new LocalExecutorConfig."""
        if pc.auto_approve:
            bash_rules = BashPermissionRules.from_dict({"*": "allow"})
            ext_rules = ExternalDirectoryRules.from_dict({"*": "allow"})
        else:
            bash_rules = BashPermissionRules()
            ext_rules = ExternalDirectoryRules()
        return LocalExecutorConfig(
            permission=PermissionBlock(bash=bash_rules, external_directory=ext_rules),
            non_interactive=pc.auto_approve,
        )

    def evaluate_bash(self, command: str) -> PermissionAction:
        """Evaluate a bash command against loaded config rules.

        Also checks session-level overrides (from interactive 'always' choices).
        Returns "allow", "ask", or "deny".
        """
        # Session-level overrides first (user chose "allow always" in this session)
        with self._lock:
            session_rules = list(self._session_allows.items())
        for pattern, action in session_rules:
            if fnmatch.fnmatch(command, pattern):
                log.debug(
                    "permission.session_match command=%r pattern=%r action=%s",
                    command[:60],
                    pattern,
                    action,
                )
                return action

        # Config rules (last-match wins)
        action = self._config.permission.bash.evaluate(command)
        if action == "ask" and self._config.non_interactive:
            return "deny"  # non-interactive mode: ask → deny

        return action

    # ── Public API ────────────────────────────────────────────────────────────

    def request(
        self, command: str, risk: RiskLevel, reasons: List[str]
    ) -> PermissionDecision:
        """Gate the command. Returns the decision."""
        while True:
            # 1. Session-cached allow (exact match or wildcard)
            if self._is_session_allowed(command):
                log.info("permission.session_allowed command=%r", command[:60])
                print(f"  {DIM}[permission] session-allowed: {command[:60]}{RESET}")
                return PermissionDecision.ALLOW_ONCE

            # 2. Config rules — the primary decision source
            action = self.evaluate_bash(command)
            if action == "allow":
                log.info("permission.config_allow command=%r", command[:60])
                return PermissionDecision.ALLOW_ONCE
            if action == "deny":
                log.info("permission.config_deny command=%r", command[:60])
                self._record_denial(command)
                return PermissionDecision.DENY

            # 3. action == "ask": check consecutive denials, then coordinate prompt ownership
            auto_deny_msg = self._check_consecutive_denials(command)
            if auto_deny_msg:
                print(f"\n  {RED}✗ {auto_deny_msg}{RESET}\n")
                return PermissionDecision.DENY

            with self._lock:
                pending = self._pending_prompts.get(command)
                if pending is None:
                    pending = _PendingPermissionPrompt(event=threading.Event())
                    self._pending_prompts[command] = pending
                    owns_prompt = True
                else:
                    owns_prompt = False

            if owns_prompt:
                try:
                    decision = self._prompt(command, risk, reasons)
                    pending.decision = decision
                finally:
                    pending.event.set()
                    with self._lock:
                        self._pending_prompts.pop(command, None)
                return pending.decision or PermissionDecision.DENY

            pending.event.wait()
            if pending.decision is not None:
                return pending.decision

    def request_external_directory(
        self, directory: str, context: str = "command"
    ) -> PermissionDecision:
        """
        Prompt the user for permission to access a directory outside the workspace.
        """
        # Auto-allow system temp dir (cross-platform: /tmp on all OSes)
        canonical = _normalize_external_path(directory)
        if canonical == "/tmp" or canonical.startswith("/tmp/"):
            return PermissionDecision.ALLOW_ONCE

        # Check config rules
        config_action = self._config.permission.external_directory.evaluate(directory)
        if config_action == "allow":
            return PermissionDecision.ALLOW_ONCE
        if config_action == "deny":
            return PermissionDecision.DENY

        # Check session cache — exact match OR child of a session-allowed parent.
        # Path resolution strips trailing slashes (/a/ -> D:/a), so startswith
        # can falsely match /a-other as a child of /a. Guard by checking the
        # next character is a path separator.
        resolved = _normalize_external_path(directory)
        with self._lock:
            for allowed in self._session_ext_allows:
                if resolved == allowed:
                    return PermissionDecision.ALLOW_ONCE
                if (
                    resolved.startswith(allowed)
                    and resolved[len(allowed) : len(allowed) + 1] == "/"
                ):
                    return PermissionDecision.ALLOW_ONCE

        # non_interactive -> deny (only after config + session checks)
        if self._config.non_interactive:
            return PermissionDecision.DENY
        if config_action == "deny":
            return PermissionDecision.DENY

        # non_interactive → deny
        if self._config.non_interactive:
            return PermissionDecision.DENY

        # Check session cache — exact match OR child of a session-allowed parent.
        # _normalize_external_path returns trailing / for dirs (e.g. "C:/a/"),
        # so resolved.startswith(allowed) correctly matches /a/sub as child of /a/.
        resolved = _normalize_external_path(directory)
        with self._lock:
            for allowed in self._session_ext_allows:
                if resolved == allowed or resolved.startswith(allowed):
                    return PermissionDecision.ALLOW_ONCE

        truncated_dir = directory if len(directory) <= 72 else directory[:69] + "..."

        print(f"\n{SEP}")
        print(f"  {BOLD}EXTERNAL DIRECTORY ACCESS{RESET}")
        print(SEP)
        print(f"  {BOLD}Directory :{RESET} {CYAN}{truncated_dir}{RESET}")
        print(f"  {BOLD}Context   :{RESET} {context}")
        print(f"  {BOLD}Warning   :{RESET} This path is OUTSIDE the workspace.")
        print(SEP)
        print(f"  {BOLD}[1]{RESET}  Allow this time")
        print(f"  {BOLD}[2]{RESET}  Allow for session (this dir + all subdirectories)")
        print(f"  {BOLD}[3]{RESET}  Deny")
        print(SEP)

        while True:
            try:
                choice = input("  Your choice [1/2/3]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {RED}✗ Denied (interrupted){RESET}\n{SEP}\n")
                return PermissionDecision.DENY

            if choice == "1":
                print(f"  {GREEN}✓ Allowed (once){RESET}\n{SEP}\n")
                return PermissionDecision.ALLOW_ONCE
            elif choice == "2":
                print(
                    f"  {GREEN}✓ Allowed (session — all commands to {truncated_dir}){RESET}\n{SEP}\n"
                )
                with self._lock:
                    self._session_ext_allows.add(canonical)
                return PermissionDecision.ALLOW_SESSION
            elif choice == "3":
                print(f"  {RED}✗ Denied{RESET}\n{SEP}\n")
                return PermissionDecision.DENY
            else:
                print("  Please enter 1, 2, or 3.")

    def add_session_allow(self, command: str) -> None:
        """Programmatically add a session-wide allow (e.g. from CLI flags)."""
        with self._lock:
            self._session_allows[command] = "allow"

    # ── Denial message for external callers (execute_shell_command) ────────────

    def denial_message(self, command: str) -> str:
        """
        Return a human-readable denial message for the given command.
        Used by execute_shell_command to craft a clear response for the LLM.
        """
        # Check stash first — populated by _check_consecutive_denials before
        # clearing history, so the specific message reaches the agent.
        with self._lock:
            if self._last_denial_message:
                msg = self._last_denial_message
                self._last_denial_message = None  # consume once
                return msg
            history = list(self._denial_history)

        threshold = self._config.consecutive_deny_threshold

        # Count only CONSECUTIVE exact matches at the END of history
        exact_consecutive = 0
        for d in reversed(history):
            if d[0] == command:
                exact_consecutive += 1
            else:
                break

        if exact_consecutive >= threshold:
            return (
                f"Permission denied for this exact command "
                f"({exact_consecutive} times). "
                f"Try a different command or approach."
            )
        return "Permission denied by user. Try a different approach."

    # ── Internal ──────────────────────────────────────────────────────────────

    def _is_session_allowed(self, command: str) -> bool:
        """Check if command matches any session-wide allow (exact or wildcard).

        Uses fnmatch for correct wildcard semantics — "git *" matches "git push"
        but NOT "github-cli". Patterns are copied under lock then matched outside
        to avoid holding the lock during iteration.
        """
        with self._lock:
            patterns = list(self._session_allows)
        for pattern in patterns:
            if fnmatch.fnmatch(command, pattern):
                return True
        return False

    def _check_consecutive_denials(self, command: str) -> Optional[str]:
        """
        If the last N denials were the same exact command string, auto-deny
        this attempt with a specific message.
        Uses config.consecutive_deny_threshold (default 2).

        NOTE: Only exact-command matching. Different commands (even same base
        like 'rm file1' vs 'rm file2') never trigger auto-deny.
        After auto-deny, clears history so the next attempt prompts again.
        """
        threshold = self._config.consecutive_deny_threshold
        with self._lock:
            if len(self._denial_history) < threshold:
                return None
            last_n = self._denial_history[-threshold:]

            if all(d[0] == command for d in last_n):
                msg = (
                    f"Auto-denied: this exact command was refused {threshold} times. "
                    f"Try a different command or approach."
                )
                self._last_denial_message = msg
                self._denial_history.clear()
                return msg

        return None

    def _record_denial(self, command: str) -> None:
        """Record a denied command in the history for consecutive-denial tracking."""
        base = _extract_command_base(command)
        with self._lock:
            self._denial_history.append((command, base))
            # Keep only last N+2 to prevent unbounded growth
            max_history = self._config.consecutive_deny_threshold + 3
            if len(self._denial_history) > max_history:
                self._denial_history = self._denial_history[-max_history:]

    def _prompt(
        self, command: str, risk: RiskLevel, reasons: List[str]
    ) -> PermissionDecision:
        """Blocking terminal prompt with 5 options."""
        reason_str = ", ".join(reasons) if reasons else "unknown risk"
        truncated_cmd = command if len(command) <= 72 else command[:69] + "..."
        base = _extract_command_base(command)

        print(f"\n{SEP}")
        print(f"  {BOLD}PERMISSION REQUEST \u2014 {risk.label()}{RESET}")
        print(SEP)
        print(f"  {BOLD}Command :{RESET} {CYAN}{truncated_cmd}{RESET}")
        print(f"  {BOLD}Risk    :{RESET} {reason_str}")
        print(SEP)
        print(f"  {BOLD}[1]{RESET}  Allow this time")
        print(f"  {BOLD}[2]{RESET}  Allow always for this session (exact command)")
        print(
            f'  {BOLD}[3]{RESET}  Allow "{base}" for entire session (all {base} commands)'
        )
        print(f"  {BOLD}[4]{RESET}  Deny")
        print(f"  {BOLD}[5]{RESET}  Deny always for this session (exact command)")
        print(SEP)

        while True:
            try:
                choice = input("  Your choice [1/2/3/4/5]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {RED}\u2717 Denied (interrupted){RESET}\n{SEP}\n")
                self._record_denial(command)
                return PermissionDecision.DENY

            if choice == "1":
                print(f"  {GREEN}\u2713 Allowed (once){RESET}\n{SEP}\n")
                return PermissionDecision.ALLOW_ONCE
            elif choice == "2":
                print(
                    f"  {GREEN}\u2713 Allowed (session \u2014 exact command){RESET}\n{SEP}\n"
                )
                with self._lock:
                    self._session_allows[command] = "allow"
                return PermissionDecision.ALLOW_SESSION
            elif choice == "3":
                wildcard = f"{base} *"
                print(
                    f'  {GREEN}\u2713 Allowed (session \u2014 all "{base}" commands){RESET}\n{SEP}\n'
                )
                with self._lock:
                    self._session_allows[wildcard] = "allow"
                return PermissionDecision.ALLOW_SESSION
            elif choice == "4":
                print(f"  {RED}\u2717 Denied{RESET}\n{SEP}\n")
                self._record_denial(command)
                return PermissionDecision.DENY
            elif choice == "5":
                print(
                    f'  {RED}\u2717 Denied (session \u2014 exact command "{command[:40]}")'
                    f"{RESET}\n{SEP}\n"
                )
                with self._lock:
                    self._session_allows[command] = "deny"
                self._record_denial(command)
                return PermissionDecision.DENY
            else:
                print("  Please enter 1, 2, 3, 4, or 5.")


# ══════════════════════════════════════════════════════════════════════════════
# § 4b  RISK ASSESSMENT MODEL
# ══════════════════════════════════════════════════════════════════════════════


class RiskAssessment(BaseModel):
    """Structured result from assess_risk(), replacing flat tuple return."""

    model_config = ConfigDict(frozen=True)

    level: RiskLevel = Field(description="Highest risk level detected.")
    reasons: List[str] = Field(
        default_factory=list, description="Human-readable risk reasons."
    )
    matched_patterns: List[str] = Field(
        default_factory=list, description="Regex patterns that matched."
    )

    @property
    def is_interactive(self) -> bool:
        return any("interactive" in r.lower() for r in self.reasons)


# ══════════════════════════════════════════════════════════════════════════════
# § 4c  ENV FILE PROTECTION
# ══════════════════════════════════════════════════════════════════════════════

ENV_FILE_PATTERNS: List[str] = [
    r"\.env(?:\s|$|['\"\`;&|])",
    r"\.env\.\w+\b",
]
_ENV_PATTERN = re.compile("|".join(ENV_FILE_PATTERNS))


def _check_env_file_exposure(command: str) -> bool:
    """Return True if command might expose .env file contents."""
    return bool(_ENV_PATTERN.search(command))


# ══════════════════════════════════════════════════════════════════════════════
# § 4d  COMMAND HANDLE (abort / cancellation)
# ══════════════════════════════════════════════════════════════════════════════


class CommandHandle:
    """Opaque handle returned by ProcessRunner for external cancellation.

    Call .cancel() from another thread to kill the process early.
    """

    def __init__(self) -> None:
        self._cancel_event = threading.Event()
        self._result_ready = threading.Event()
        self._result: Optional[CommandResult] = None

    def cancel(self, wait: bool = False, timeout: float = 2.0) -> None:
        """Signal the running command to stop. Thread-safe.

        Args:
            wait: If True, block until the command finishes and return partial output.
                  If False (default), return immediately (existing behavior).
            timeout: Maximum seconds to wait when wait=True.
        """
        self._cancel_event.set()
        if wait:
            self._result_ready.wait(timeout=timeout)

    def wait(self, timeout: Optional[float] = None) -> Optional[CommandResult]:
        """Block until command finishes or timeout. Returns CommandResult or None."""
        self._result_ready.wait(timeout=timeout)
        return self._result


# ══════════════════════════════════════════════════════════════════════════════
# § 4e  DOOM LOOP DETECTOR
# ══════════════════════════════════════════════════════════════════════════════


class DoomLoopDetector:
    """Detects when the agent is stuck repeating the same command."""

    def __init__(self, threshold: int = 3) -> None:
        self._threshold = threshold
        self._history: List[str] = []
        self._lock = threading.Lock()

    def record(self, command: str) -> bool:
        """Record a command. Returns True if doom loop detected."""
        with self._lock:
            self._history.append(command)
            if len(self._history) < self._threshold:
                return False
            last_n = self._history[-self._threshold :]
            return all(c == command for c in last_n)

    def reset(self) -> None:
        with self._lock:
            self._history.clear()


# ══════════════════════════════════════════════════════════════════════════════
# § 4f  CONFIG WATCHER (hot-reload)
# ══════════════════════════════════════════════════════════════════════════════


class ConfigWatcher:
    """Watches the project config file and triggers reload on change.

    TODO: Wire into LocalExecutorToolkit.__init__() for hot-reload support.
    Currently unused — instantiated nowhere.
    """

    def __init__(
        self,
        loader: ConfigLoader,
        on_reload: Callable[[LocalExecutorConfig], None],
    ) -> None:
        self._loader = loader
        self._on_reload = on_reload
        self._path = loader.project_config_path()
        self._mtime: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _watch(self) -> None:
        while not self._stop.is_set():
            try:
                if self._path.exists():
                    mtime = self._path.stat().st_mtime
                    if self._mtime is None:
                        self._mtime = mtime
                        config = self._loader.load()
                        self._on_reload(config)
                        log.info(
                            "Config loaded from newly detected file %s", self._path
                        )
                    elif mtime != self._mtime:
                        config = self._loader.load()
                        self._on_reload(config)
                        log.info("Config reloaded from %s", self._path)
                        self._mtime = mtime
            except Exception as exc:
                log.warning("Config watcher error: %s", exc)
            self._stop.wait(timeout=2.0)


# ══════════════════════════════════════════════════════════════════════════════
# § 4g  SCHEMA GENERATION
# ══════════════════════════════════════════════════════════════════════════════


def generate_json_schema(output_path: Optional[str] = None) -> dict:
    """Generate and optionally write the JSON schema for local_executor.json."""
    permission_action_enum = {
        "type": "string",
        "enum": ["allow", "ask", "deny"],
    }
    wildcard_rule_map = {
        "type": "object",
        "propertyNames": {"type": "string"},
        "additionalProperties": permission_action_enum,
        "default": {},
    }
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "LocalExecutorConfigFile",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "$schema": {"type": "string"},
            "shell": {"type": ["string", "null"]},
            "timeout_ms": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_TIMEOUT_MS,
                "default": 120000,
            },
            "permission": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "*": permission_action_enum,
                    "bash": wildcard_rule_map,
                    "external_directory": wildcard_rule_map,
                },
                "default": {
                    "*": "ask",
                    "bash": {},
                    "external_directory": {},
                },
            },
            "consecutive_deny_threshold": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "default": 2,
            },
            "non_interactive": {"type": "boolean", "default": False},
        },
    }
    if output_path:
        Path(output_path).write_text(_json.dumps(schema, indent=2), encoding="utf-8")
        log.info("Schema written to %s", output_path)
    return schema


def _tokenize_command(command: str) -> List[str]:
    """Split shell text into tokens with a regex fallback for malformed syntax."""
    try:
        return shlex.split(command)
    except ValueError:
        return re.findall(r'(?:[^\s"\']+|"[^"]*"|\'[^\']*\')+', command)


def _clean_command_token(token: str) -> str:
    """Strip wrappers/operators around a token before path analysis."""
    token = re.sub(r"^(['\"])(.*)\1$", r"\2", token)
    token = re.sub(r"[;,|&]+$", "", token)
    return _normalize_bash_path(token)


def _extract_command_paths(
    command: str,
    cwd: str,
    include_relative: bool = False,
) -> List[Tuple[str, str]]:
    """Return (original_token, resolved_path) pairs for path-like command tokens."""
    base_cwd = Path(cwd).resolve()
    result: List[Tuple[str, str]] = []
    for raw_token in _tokenize_command(command):
        token = _clean_command_token(raw_token)
        if not token:
            continue
        if os.path.isabs(token):
            candidate = Path(token)
        elif include_relative:
            candidate = base_cwd / token
        else:
            continue
        try:
            result.append((token, str(candidate.resolve())))
        except (OSError, ValueError):
            pass
    return result


def _normalize_bash_path(p: str) -> str:
    """
    Normalize Git Bash / MSYS style paths to Windows paths.

    Git Bash uses /c/, /d/, etc. which Python's Path.resolve() doesn't handle.
    Converts:
      /c/Users/...  →  C:/Users/...
      /cygdrive/c/... → C:/Users/...
      /mnt/c/...    →  C:/Users/...
    """
    if not IS_WINDOWS:
        return p
    # /c/... → C:/...
    m = re.match(r"^/([a-zA-Z])(/.*)$", p)
    if m:
        return f"{m.group(1).upper()}:{m.group(2)}"
    # /cygdrive/c/... → C:/...
    m = re.match(r"^/cygdrive/([a-zA-Z])(/.*)$", p)
    if m:
        return f"{m.group(1).upper()}:{m.group(2)}"
    # /mnt/c/... → C:/...
    m = re.match(r"^/mnt/([a-zA-Z])(/.*)$", p)
    if m:
        return f"{m.group(1).upper()}:{m.group(2)}"
    return p


# ══════════════════════════════════════════════════════════════════════════════
# § 5  EXTERNAL PATH DETECTION
# ══════════════════════════════════════════════════════════════════════════════


def detect_external_paths(command: str, cwd: str) -> List[str]:
    """
    Advisory scan: find absolute paths in command arguments that reference
    locations outside cwd. Returns list of external directory strings.

    This is advisory only — it does NOT block execution. Mirrors OpenCode's
    externalCommandDirectories() function.
    """
    cwd_resolved = str(Path(cwd).resolve())
    external_order: Dict[str, None] = {}

    for _orig_token, resolved in _extract_command_paths(
        command, cwd, include_relative=False
    ):
        try:
            # If it's inside cwd, not external
            if resolved.startswith(cwd_resolved + os.sep) or resolved == cwd_resolved:
                continue
            parent = str(Path(resolved).parent.resolve())
            external_order[parent] = None  # preserve insertion order, deduplicate
        except (OSError, ValueError):
            pass

    return list(external_order.keys())


def contains_path(parent: str, child: str) -> bool:
    """
    Check if `child` is inside `parent` directory.

    Mirrors OpenCode's FSUtil.contains():
      - Resolves both paths to canonical form (follows symlinks)
      - Uses path.relative() to determine containment
      - Returns True if child == parent or child is a descendant of parent
    """
    try:
        parent_resolved = str(Path(parent).resolve())
        child_resolved = str(Path(child).resolve())
        rel = os.path.relpath(child_resolved, parent_resolved)
        return rel == "." or (not os.path.isabs(rel) and not rel.startswith(".."))
    except (OSError, ValueError):
        return False


def contains_path_lexical(parent: str, child: str) -> bool:
    """
    Check if `child` is lexically inside `parent` WITHOUT following symlinks.

    Unlike contains_path(), this uses the raw string paths without resolve(),
    so a symlink at `parent/link` pointing to `/etc` is still considered
    "inside" `parent` for the purpose of detecting symlink escapes.

    Mirrors OpenCode's FSUtil.contains() used for the lexical check in
    LocationMutation.resolve() before symlink resolution.
    """
    try:
        # Normalize to absolute paths without following symlinks.
        # os.path.abspath resolves relative to cwd but does NOT follow symlinks.
        # If child is relative, resolve it against parent (not cwd) so that
        # relative paths like "subdir" work correctly when parent is absolute.
        parent_abs = os.path.abspath(parent)
        if os.path.isabs(child):
            child_abs = os.path.abspath(child)
        else:
            child_abs = os.path.abspath(os.path.join(parent_abs, child))
        rel = os.path.relpath(child_abs, parent_abs)
        return rel == "." or (not os.path.isabs(rel) and not rel.startswith(".."))
    except (OSError, ValueError):
        return False


def detect_symlink_escape(path: str, workspace_dir: str) -> Optional[Tuple[str, str]]:
    """
    Check if `path` is a symlink inside the workspace that resolves outside it.

    Mirrors OpenCode's LocationEscape detection:
      1. Lexical check: is the path (before symlink resolution) inside the
         workspace? Uses contains_path_lexical() which does NOT follow symlinks.
      2. If lexically inside, check if it's actually a symlink.
      3. Follow the symlink to get the real path.
      4. Canonical check: does the real path escape the workspace?

    Returns (real_path, escaped_from) if symlink escape detected, else None.
    """
    try:
        p = Path(path)
        if not p.is_absolute():
            p = Path(workspace_dir) / p

        # Step 1: Lexical containment check (no symlink resolution)
        # This is the key fix — use absolute() not resolve() so symlinks
        # that are lexically inside the workspace are caught.
        if not contains_path_lexical(workspace_dir, str(p)):
            return None  # Not lexically inside — handled as external path, not symlink escape

        # Step 2: Is it actually a symlink?
        if not p.is_symlink():
            return None

        # Step 3: Follow the symlink
        real = str(p.resolve())

        # Step 4: Canonical containment check — does the real path escape?
        if not contains_path(workspace_dir, real):
            return real, str(p)

    except (OSError, ValueError):
        pass

    return None


def _resolve_workdir(
    workdir: Optional[str],
    workspace_dir: str,
    cwd_state: str,
) -> Tuple[str, Optional[str]]:
    """
    Resolve the working directory for a command, always returning a valid path.

    Resolution order:
      1. If workdir is provided:
         - Absolute  → resolve and check containment in workspace
         - Relative  → resolve against cwd_state, then check containment
      2. If workdir is None → use cwd_state (session tracked directory)

    Returns (resolved_path, external_directory_or_None).
    If the resolved path is outside the workspace, external_directory is set
    to the parent directory that the caller must get permission for.
    """
    if workdir:
        # Normalize Git Bash paths (/c/...) before resolving
        workdir = _normalize_bash_path(workdir)
        if os.path.isabs(workdir):
            resolved = str(Path(workdir).resolve())
        else:
            resolved = str((Path(cwd_state) / workdir).resolve())
    else:
        resolved = cwd_state

    if contains_path(workspace_dir, resolved):
        return resolved, None

    return resolved, str(Path(resolved).resolve())


def extract_external_command_paths(command: str, cwd: str) -> List[str]:
    """
    Parse command arguments and return absolute paths that are outside cwd.

    This is more aggressive than detect_external_paths() — it returns the
    actual external paths (not just parent directories) for permission checking.
    """
    cwd_resolved = str(Path(cwd).resolve())
    external: List[str] = []

    for _orig_token, resolved in _extract_command_paths(
        command, cwd, include_relative=False
    ):
        try:
            if not contains_path(cwd_resolved, resolved):
                external.append(resolved)
        except (OSError, ValueError):
            pass

    return external


def _extract_command_tokens(
    command: str,
    cwd: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """
    Parse command and return (original_token, resolved_path) for each argument.

    Unlike extract_external_command_paths which only returns resolved paths,
    this preserves the original token so callers can check for symlink escapes
    on the unresolved path.
    """
    return _extract_command_paths(command, cwd or os.getcwd(), include_relative=True)


# ══════════════════════════════════════════════════════════════════════════════
# § 6  PROCESS RUNNER
# ══════════════════════════════════════════════════════════════════════════════


class ProcessRunner:
    """
    Spawns and manages child processes, mirroring OpenCode's AppProcess layer.

    Features
    ────────
    • Combined stdout + stderr (combineOutput: true in OpenCode)
    • 1 MB output cap with truncation tracking
    • Configurable timeout with SIGTERM → SIGKILL escalation
    • Optional stdin injection for non-interactive commands
    • New process session on POSIX for clean group kill
    • taskkill /T /F on Windows for tree kill
    """

    def __init__(self, shell: Optional[str] = None):
        self.shell = shell or _default_shell()

    def run(
        self,
        command: str,
        cwd: str,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        combine_output: bool = True,
        on_output: Optional[Callable[[str], None]] = None,
        # TODO: Wire on_output into execute_shell_command for live streaming
        # (currently hardcoded to None in the toolkit). Would give long builds
        # incremental progress display — a feature OpenCode also lacks.
        stdin: Optional[str] = None,
        handle: Optional[CommandHandle] = None,
    ) -> CommandResult:
        """Execute command and return a structured CommandResult.

        Args:
            command: Shell command string.
            cwd: Working directory.
            timeout_ms: Timeout in milliseconds.
            combine_output: If True (default), merge stdout+stderr into a single
                           output string. If False, capture them separately and
                           populate the stdout/stderr fields on CommandResult.
            on_output: Optional callback called with each decoded line as it arrives.
            stdin: Optional UTF-8 text written to the process stdin, then closed.
            handle: Optional CommandHandle for external cancellation support.
        """
        timeout_ms = max(1, min(timeout_ms, MAX_TIMEOUT_MS))
        timeout_s = timeout_ms / 1_000.0

        # Validate working directory
        cwd_path = Path(cwd)
        if not cwd_path.exists():
            return CommandResult(
                command=command,
                exit_code=1,
                timed_out=False,
                truncated=False,
                cwd=cwd,
                output=f"Error: working directory does not exist: {cwd}",
            )
        if not cwd_path.is_dir():
            return CommandResult(
                command=command,
                exit_code=1,
                timed_out=False,
                truncated=False,
                cwd=cwd,
                output=f"Error: working directory is not a directory: {cwd}",
            )

        # Advisory: external path warnings
        ext_warnings = [
            f"References external path '{d}' — command runs with host filesystem access"
            for d in detect_external_paths(command, cwd)
        ]

        # Spawn — use _shell_argv to pick the right shell and flag (-c vs /c)
        # On Windows with shell=True Python always uses cmd.exe ignoring
        # executable=, so we explicitly build the argv list and use shell=False.
        # On POSIX we still use shell=True + start_new_session for clean group kill.
        # Runtime fallback: if the primary shell fails to spawn on Windows,
        # retry with cmd.exe before giving up.
        argv = _shell_argv(self.shell, command)
        fallback_argv: Optional[List[str]] = None
        if IS_WINDOWS and not _shell_is_cmd(self.shell):
            fallback_argv = _shell_argv(os.environ.get("COMSPEC", "cmd.exe"), command)

        proc: Optional[subprocess.Popen] = None
        stderr_mode = subprocess.STDOUT if combine_output else subprocess.PIPE
        try:
            if IS_WINDOWS:
                proc = subprocess.Popen(
                    argv,
                    shell=False,
                    cwd=str(cwd_path),
                    stdout=subprocess.PIPE,
                    stderr=stderr_mode,
                    stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            else:
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    executable=self.shell,
                    cwd=str(cwd_path),
                    stdout=subprocess.PIPE,
                    stderr=stderr_mode,
                    stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                    start_new_session=True,
                )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            # Primary shell failed — try cmd.exe fallback on Windows
            if fallback_argv:
                log.warning(
                    "Primary shell %s failed to spawn (%s), retrying with cmd.exe",
                    self.shell,
                    exc,
                )
                try:
                    proc = subprocess.Popen(
                        fallback_argv,
                        shell=False,
                        cwd=str(cwd_path),
                        stdout=subprocess.PIPE,
                        stderr=stderr_mode,
                        stdin=subprocess.PIPE
                        if stdin is not None
                        else subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                except Exception as fallback_exc:
                    return CommandResult(
                        command=command,
                        exit_code=1,
                        timed_out=False,
                        truncated=False,
                        cwd=cwd,
                        output=(
                            f"Failed to start process with primary shell ({exc}) "
                            f"and cmd.exe fallback ({fallback_exc})"
                        ),
                        warnings=ext_warnings,
                    )
            else:
                return CommandResult(
                    command=command,
                    exit_code=1,
                    timed_out=False,
                    truncated=False,
                    cwd=cwd,
                    output=f"Failed to start process: {exc}",
                    warnings=ext_warnings,
                )

        # Output collector (runs in background thread to prevent pipe blocking)
        chunks: List[bytes] = []
        stderr_chunks: List[bytes] = []
        total_in: int = 0
        truncated: bool = False
        stderr_total_in: int = 0
        stderr_truncated: bool = False
        read_done = threading.Event()
        stderr_read_done = threading.Event()
        stdin_done = threading.Event()

        def _reader() -> None:
            nonlocal total_in, truncated
            assert proc.stdout is not None
            buffer = b""
            try:
                while True:
                    chunk = proc.stdout.read(16_384)
                    if not chunk:
                        if buffer and on_output:
                            on_output(buffer.decode("utf-8", errors="replace"))
                        break
                    remaining = MAX_CAPTURE_BYTES - total_in
                    if remaining > 0:
                        keep = chunk if len(chunk) <= remaining else chunk[:remaining]
                        chunks.append(keep)
                    else:
                        truncated = True
                    total_in += len(chunk)
                    if total_in > MAX_CAPTURE_BYTES:
                        truncated = True
                    if on_output:
                        buffer += chunk
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            on_output(line.decode("utf-8", errors="replace") + "\n")
            except Exception as exc:
                log.warning("Output reader error: %s", exc)
            finally:
                read_done.set()

        def _stderr_reader() -> None:
            """Read stderr when combine_output=False, with byte limit."""
            nonlocal stderr_total_in, stderr_truncated
            assert proc.stderr is not None
            try:
                while True:
                    chunk = proc.stderr.read(16_384)
                    if not chunk:
                        break
                    remaining = MAX_CAPTURE_BYTES - stderr_total_in
                    if remaining > 0:
                        keep = chunk if len(chunk) <= remaining else chunk[:remaining]
                        stderr_chunks.append(keep)
                    else:
                        stderr_truncated = True
                    stderr_total_in += len(chunk)
                    if stderr_total_in > MAX_CAPTURE_BYTES:
                        stderr_truncated = True
            except Exception as exc:
                log.warning("Stderr reader error: %s", exc)
            finally:
                stderr_read_done.set()

        def _stdin_writer() -> None:
            try:
                if stdin is None or proc.stdin is None:
                    return
                proc.stdin.write(stdin.encode("utf-8"))
                proc.stdin.close()
            except Exception as exc:
                log.warning("Stdin writer error: %s", exc)
            finally:
                stdin_done.set()

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        stderr_reader: Optional[threading.Thread] = None
        if not combine_output and proc.stderr is not None:
            stderr_reader = threading.Thread(target=_stderr_reader, daemon=True)
            stderr_reader.start()

        stdin_writer: Optional[threading.Thread] = None
        if stdin is not None and proc.stdin is not None:
            stdin_writer = threading.Thread(target=_stdin_writer, daemon=True)
            stdin_writer.start()
        else:
            stdin_done.set()

        # Wait for process with timeout + cancellation support
        timed_out = False
        deadline = time.monotonic() + timeout_s
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    self._kill(proc)
                    break
                if handle and handle._cancel_event.is_set():
                    self._kill(proc)
                    result = CommandResult(
                        command=command,
                        exit_code=None,
                        output="Command cancelled by caller.",
                        truncated=False,
                        timed_out=False,
                        cwd=cwd,
                        denied=False,
                    )
                    if handle:
                        handle._result = result
                        handle._result_ready.set()
                    return result
                try:
                    proc.wait(timeout=min(0.2, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except KeyboardInterrupt:
            self._kill(proc)
            timed_out = True

        # Wait for readers to flush (up to 10 s after process exits; process
        # termination itself may already have spent additional time in _kill()).
        read_done.wait(timeout=10.0)
        reader.join(timeout=5.0)
        if reader.is_alive():
            log.warning(
                "Output reader thread still alive after 5s — output may be incomplete"
            )
        stderr_read_done.wait(timeout=10.0)
        if stderr_reader is not None:
            stderr_reader.join(timeout=5.0)
            if stderr_reader.is_alive():
                log.warning(
                    "Stderr reader thread still alive after 5s — output may be incomplete"
                )
        stdin_done.wait(timeout=2.0)
        if stdin_writer is not None:
            stdin_writer.join(timeout=1.0)

        exit_code = proc.returncode if not timed_out else None
        stdout_str = b"".join(chunks).decode("utf-8", errors="replace")
        stderr_str = (
            b"".join(stderr_chunks).decode("utf-8", errors="replace")
            if stderr_chunks
            else None
        )

        # When combine_output=True, output includes both streams (stderr was merged by OS)
        # When combine_output=False, output is stdout only, stderr is separate
        output = stdout_str

        # Save full output to temp file when truncation happened (line or byte cap)
        line_count = len(output.splitlines())
        output_file = (
            _save_output_to_file(output, command)
            if line_count > MAX_OUTPUT_LINES or truncated
            else None
        )

        # Save stderr separately when truncated (combine_output=False)
        stderr_output_file: Optional[str] = None
        if stderr_str is not None and stderr_truncated:
            stderr_output_file = _save_output_to_file(stderr_str, command)

        result = CommandResult(
            command=command,
            exit_code=exit_code,
            output=output,
            truncated=truncated,
            timed_out=timed_out,
            cwd=cwd,
            warnings=ext_warnings,
            output_file=output_file,
            stderr_output_file=stderr_output_file,
            stdout=stdout_str if not combine_output else None,
            stderr=stderr_str if not combine_output else None,
            stderr_truncated=stderr_truncated if not combine_output else False,
        )
        if handle:
            handle._result = result
            handle._result_ready.set()
        return result

    def _kill(self, proc: subprocess.Popen) -> None:
        """
        Graceful termination sequence:
          POSIX  → kill process group with SIGTERM, wait 3 s, then SIGKILL
          Windows → taskkill /pid /T /F  (kills entire process tree)
        """
        if IS_WINDOWS:
            self._kill_windows(proc)
        else:
            self._kill_posix(proc)

    def _kill_windows(self, proc: subprocess.Popen) -> None:
        """Kill process tree on Windows using taskkill."""
        try:
            subprocess.run(
                ["taskkill", "/pid", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _kill_posix(self, proc: subprocess.Popen) -> None:
        """Kill process group on POSIX with SIGTERM → SIGKILL escalation."""
        # Graceful: SIGTERM
        try:
            pgid = os.getpgid(proc.pid)  # type: ignore[attr-defined]
            os.killpg(pgid, signal.SIGTERM)  # type: ignore[attr-defined]
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.terminate()
            except Exception:
                pass

        # Wait for graceful shutdown
        try:
            proc.wait(timeout=FORCE_KILL_DELAY_S)
            return
        except subprocess.TimeoutExpired:
            pass

        # Escalate: SIGKILL
        try:
            pgid = os.getpgid(proc.pid)  # type: ignore[attr-defined]
            os.killpg(pgid, signal.SIGKILL)  # type: ignore[attr-defined]
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass

        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# § 7  WORKING DIRECTORY TRACKER
# ══════════════════════════════════════════════════════════════════════════════

_CD_RE = re.compile(r"^\s*cd\s+(.+?)\s*$")


def parse_cd_target(command: str, current_cwd: str) -> Optional[str]:
    """
    If command is purely `cd <path>`, return the new resolved directory.
    Returns None for compound commands or relative-but-unresolvable paths.
    """
    m = _CD_RE.match(command.strip())
    if not m:
        return None

    target = m.group(1).strip()
    # Strip matching outer quotes (but not unmatched ones)
    if (target.startswith('"') and target.endswith('"')) or (
        target.startswith("'") and target.endswith("'")
    ):
        target = target[1:-1]

    # Reject compound commands: cd /foo && ls, cd /foo; ls, cd /foo | something
    if re.search(r"[;&|]", target):
        return None

    if target in ("", "-"):
        return None  # can't reliably track `cd -` or bare `cd`
    if target == "~":
        return str(Path.home())
    if os.path.isabs(target):
        return str(Path(target).resolve())
    return str((Path(current_cwd) / target).resolve())


# ══════════════════════════════════════════════════════════════════════════════
# § 8  TOOLKIT  (public API)
# ══════════════════════════════════════════════════════════════════════════════


class LocalExecutorToolkit:
    """
    Factory that produces typed ADK-compatible tool functions.

    Usage
    ─────
    toolkit = LocalExecutorToolkit(
        workspace_dir=".",
        permission_config=PermissionConfig(
            min_ask_level=RiskLevel.MODERATE,
            deny_critical=False,
        ),
    )
    execute_shell_command = toolkit.get_tool_function()
    # or
    execute_shell_command, run_python_code = toolkit.get_tool_functions()

    Pass these directly into LlmAgent(tools=[...]).
    """

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        shell: Optional[str] = None,
        permission_config: Optional[PermissionConfig] = None,
        config: Optional[LocalExecutorConfig] = None,
        load_config_files: bool = True,
    ):
        self.workspace_dir = str(Path(workspace_dir or os.getcwd()).resolve())
        if permission_config is not None:
            warnings.warn(
                "permission_config is deprecated; prefer LocalExecutorConfig.",
                DeprecationWarning,
                stacklevel=2,
            )
        if config is not None and permission_config is not None:
            log.warning(
                "permission_config was provided alongside config and will be ignored."
            )

        # Config resolution: explicit > file-loaded > backward-compat PermissionConfig
        if config is not None:
            self._config = config
        elif load_config_files:
            loader = ConfigLoader(self.workspace_dir)
            self._config = loader.load()
        else:
            self._config = LocalExecutorConfig()

        # Backward compat: if old-style PermissionConfig is passed, use it
        self._compat_permission_config = permission_config

        effective_shell = shell or self._config.shell or _default_shell()
        self._runner = ProcessRunner(shell=effective_shell)
        self._permissions = PermissionManager(
            config=self._config,
            compat=self._compat_permission_config,
        )
        self._cwd = self.workspace_dir

    # ── Primary tool: execute_shell_command ───────────────────────────────────

    def get_tool_function(self) -> Callable:
        """Return the primary execute_shell_command tool function."""
        runner = self._runner
        permissions = self._permissions
        cwd_state = [self.workspace_dir]  # mutable single-element list
        call_count = [0]  # mutable counter for throttled cleanup
        doom_detector = DoomLoopDetector(threshold=3)
        state_lock = threading.Lock()

        def execute_shell_command(
            command: str,
            workdir: Optional[str] = None,
            timeout_ms: int = DEFAULT_TIMEOUT_MS,
            stdin: Optional[str] = None,
            combine_output: bool = True,
        ) -> dict:
            """Execute a shell command with permission checking and structured output.

            This is the primary execution tool. It handles all shell operations
            including running Python code, file manipulation, git commands,
            package management, and system inspection.

            For Python code, use standard shell syntax:
              • Single line : python3 -c "print('hello')"
              • Script file : python3 /path/to/script.py
              • Pipe input  : echo "print(1+1)" | python3

            Working directory persists across calls — a standalone `cd /path`
            command updates the session working directory.

            Args:
                command:    Shell command string to execute. Supports full shell
                            syntax: pipes, redirects, &&, ||, subshells, etc.
                workdir:    Override the working directory for this call only.
                            Absolute path, or relative to current session cwd.
                            Defaults to the current session working directory.
                timeout_ms: Timeout in milliseconds.
                            Default: 120000 (2 min). Maximum: 600000 (10 min).
                            If the process exceeds this, it is killed and
                            timed_out=True is returned. Retry with a higher
                            value for long-running builds or downloads.
                stdin:      Optional UTF-8 text piped to the command's stdin.
                            Useful for non-interactive tools that read from stdin.
                combine_output: If True (default), merge stdout+stderr into a single
                            output string. If False, capture them separately and
                            label them as [stdout]/[stderr] in the output.

            Returns:
                dict with:
                  output       — combined stdout + stderr (string)
                  exit_code    — integer, or null on timeout
                  success      — True if exit_code == 0
                  truncated    — True if output exceeded 1 MB and was cut
                  timed_out    — True if process was killed for timeout
                  denied       — True if permission was refused
                  working_dir  — absolute path the command ran in
                  warnings     — advisory notes (e.g. external path refs)
                  summary      — one-line status ("Exit code: 0." etc.)
            """
            # ── Check for symlink escape BEFORE resolving workdir ──────────
            # Must check on the raw (unresolved) path so lexical containment
            # catches symlinks inside workspace pointing outside.
            with state_lock:
                current_cwd = cwd_state[0]
            raw_workdir = workdir or current_cwd
            symlink_info = detect_symlink_escape(raw_workdir, self.workspace_dir)
            if symlink_info:
                real_path, symlink_path = symlink_info
                return CommandResult(
                    command=command,
                    exit_code=1,
                    output=(
                        f"Symlink escape detected: '{symlink_path}' is a symlink "
                        f"inside the workspace that resolves to '{real_path}', "
                        f"which is outside the workspace.\n"
                        f"Use the real path directly or remove the symlink."
                    ),
                    truncated=False,
                    timed_out=False,
                    cwd=raw_workdir,
                    denied=True,
                ).to_dict()

            # ── Resolve working directory (NEVER None — always a real path) ──
            resolved_cwd, external_workdir = _resolve_workdir(
                workdir, self.workspace_dir, current_cwd
            )

            # ── External workdir: require user permission ─────────────────
            if external_workdir:
                ext_decision = permissions.request_external_directory(
                    external_workdir,
                    context=f"working directory for command: {command[:60]}",
                )
                if ext_decision == PermissionDecision.DENY:
                    return CommandResult(
                        command=command,
                        exit_code=126,
                        output=(
                            f"Permission denied: working directory '{external_workdir}' "
                            f"is outside the workspace '{self.workspace_dir}'."
                        ),
                        truncated=False,
                        timed_out=False,
                        cwd=resolved_cwd,
                        denied=True,
                    ).to_dict()

            # ── Throttled cleanup of old output files ─────────────────────
            with state_lock:
                call_count[0] += 1
                current_call_count = call_count[0]
            if current_call_count % 50 == 0:
                _cleanup_old_outputs()

            # ── Clamp timeout ──────────────────────────────────────────────
            timeout_ms = max(1, min(int(timeout_ms), MAX_TIMEOUT_MS))

            # ── Interactive command guard ──────────────────────────────────
            interactive, int_reason = is_interactive(command)
            if interactive:
                return CommandResult(
                    command=command,
                    exit_code=1,
                    output=(
                        f"Refused: '{command.strip()}' appears to be an interactive "
                        f"command ({int_reason}).\n"
                        "Interactive programs require a terminal and would hang.\n"
                        "Use non-interactive alternatives, e.g.:\n"
                        '  • python3 -c "..."  instead of  python3\n'
                        "  • git log --oneline  instead of  git log (pager)\n"
                        "  • cat file           instead of  less file"
                    ),
                    truncated=False,
                    timed_out=False,
                    cwd=resolved_cwd,
                    denied=True,
                ).to_dict()

            # ── External path scan in command arguments ────────────────────
            # Check if the command references absolute paths outside the workspace.
            # If so, prompt the user for permission (not just advisory).
            # Use original tokens (not resolved) for symlink escape detection.
            cmd_tokens = _extract_command_tokens(command, resolved_cwd)
            checked_ext_dirs: Set[str] = set()
            for orig_token, resolved_token in cmd_tokens:
                if not os.path.isabs(resolved_token):
                    continue
                if contains_path(resolved_cwd, resolved_token):
                    continue

                # Check for symlink escape using the ORIGINAL unresolved token
                symlink_info = detect_symlink_escape(orig_token, self.workspace_dir)
                if symlink_info:
                    real_path, symlink_path = symlink_info
                    return CommandResult(
                        command=command,
                        exit_code=1,
                        output=(
                            f"Symlink escape detected: '{symlink_path}' is a symlink "
                            f"inside the workspace that resolves to '{real_path}', "
                            f"which is outside the workspace.\n"
                            f"Use the real path directly or remove the symlink."
                        ),
                        truncated=False,
                        timed_out=False,
                        cwd=resolved_cwd,
                        denied=True,
                    ).to_dict()

                # Prompt for external directory permission (deduplicate by parent dir)
                ext_dir = str(Path(resolved_token).parent.resolve())
                if ext_dir in checked_ext_dirs:
                    continue
                checked_ext_dirs.add(ext_dir)

                ext_decision = permissions.request_external_directory(
                    ext_dir,
                    context=f"referenced in command: {command[:60]}",
                )
                if ext_decision == PermissionDecision.DENY:
                    return CommandResult(
                        command=command,
                        exit_code=126,
                        output=(
                            f"Permission denied: command references external path "
                            f"'{resolved_token}' outside the workspace."
                        ),
                        truncated=False,
                        timed_out=False,
                        cwd=resolved_cwd,
                        denied=True,
                    ).to_dict()

            # ── Risk assessment + permission ───────────────────────────────
            risk, reasons = assess_risk(command)

            # ── Env file exposure warning ────────────────────────────────
            if _check_env_file_exposure(command):
                log.warning("Env file exposure detected in command: %s", command[:80])
                if "env" not in reasons:
                    reasons.append("possible .env file exposure")

            decision = permissions.request(command, risk, reasons)

            if decision == PermissionDecision.DENY:
                deny_msg = permissions.denial_message(command)
                return CommandResult(
                    command=command,
                    exit_code=126,  # POSIX: command cannot execute
                    output=deny_msg,
                    truncated=False,
                    timed_out=False,
                    cwd=resolved_cwd,
                    denied=True,
                ).to_dict()

            # ── Doom loop detection ──────────────────────────────────────
            if doom_detector.record(command):
                return CommandResult(
                    command=command,
                    exit_code=126,
                    output=(
                        f"Doom loop detected: the same command '{command[:60]}' has been run "
                        f"{doom_detector._threshold} times consecutively. "
                        "Stop repeating this command. Investigate why it keeps failing and try a "
                        "different approach, or ask the user for guidance."
                    ),
                    truncated=False,
                    timed_out=False,
                    cwd=resolved_cwd,
                    denied=True,
                ).to_dict()

            # ── Execute ────────────────────────────────────────────────────
            result = runner.run(
                command,
                resolved_cwd,
                timeout_ms,
                combine_output=combine_output,
                on_output=None,
                stdin=stdin,
            )

            # ── Track cwd if this was a bare `cd` ─────────────────────────
            if result.exit_code == 0:
                doom_detector.reset()
                new_cwd = parse_cd_target(command, resolved_cwd)
                if new_cwd and Path(new_cwd).is_dir():
                    # Only allow cd within the workspace
                    if contains_path(self.workspace_dir, new_cwd):
                        with state_lock:
                            cwd_state[0] = new_cwd
                    else:
                        with state_lock:
                            cwd_state[0] = self.workspace_dir
                        # cd outside workspace — warn but still track
                        print(
                            f"  {YELLOW}⚠ Warning: cd to '{new_cwd}' is outside "
                            f"workspace. Future commands will use the workspace "
                            f"root '{self.workspace_dir}'.{RESET}"
                        )

            # cwd_state update is safe because ADK calls tools sequentially.
            # If concurrent tool calls are ever supported, this must be atomic.
            return result.to_dict()

        return execute_shell_command

    # ── Convenience wrapper: run_python_code ─────────────────────────────────

    def get_python_tool(self) -> Callable:
        """
        Return a run_python_code convenience tool.

        Writes multi-line Python source to a temp file and executes it.
        This is a thin wrapper over execute_shell_command — for most
        cases prefer passing python3 -c "..." directly to the shell tool.
        """
        execute = self.get_tool_function()
        workspace = self.workspace_dir

        def run_python_code(
            code: str,
            workdir: Optional[str] = None,
            timeout_ms: int = DEFAULT_TIMEOUT_MS,
        ) -> dict:
            """Run multi-line Python code by writing it to a temporary script file.

            Prefer this over python3 -c "..." for multi-line scripts where
            escaping becomes awkward. Uses the same shell execution harness
            as execute_shell_command.

            Args:
                code:       Python source code (multi-line strings fine).
                workdir:    Working directory for execution.
                timeout_ms: Timeout in milliseconds (default 120000, max 600000).

            Returns:
                Same dict structure as execute_shell_command.
            """
            # Write code to a named temp file (deleted after execution)
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".py",
                    prefix="agent_script_",
                    delete=False,
                    dir=workdir or workspace,
                ) as fh:
                    fh.write(code)
                    tmp_path = fh.name
            except OSError as exc:
                return CommandResult(
                    command="<python_script>",
                    exit_code=1,
                    output=f"Could not create temp script: {exc}",
                    truncated=False,
                    timed_out=False,
                    cwd=workdir or workspace,
                ).to_dict()

            python_bin = shlex.quote(sys.executable or "python3")
            cmd = f"{python_bin} {shlex.quote(tmp_path)}"

            try:
                return execute(cmd, workdir=workdir, timeout_ms=timeout_ms)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        return run_python_code

    # ── Combined getter (backwards compat) ────────────────────────────────────

    def get_tool_functions(self) -> Tuple[Callable, Callable]:
        """
        Return (execute_shell_command, run_python_code).

        Both are ADK-compatible tool functions with docstrings and type hints.
        Pass them directly to LlmAgent(tools=[...]).
        """
        return self.get_tool_function(), self.get_python_tool()


# ══════════════════════════════════════════════════════════════════════════════
# § 10  PUBLIC EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Toolkit (recommended entry point)
    "LocalExecutorToolkit",
    # Pydantic models (OpenCode-compatible)
    "BashInput",
    "BashOutput",
    "BashStructuredOutput",
    "Outcome",
    # Process runner
    "ProcessRunner",
    # Data models
    "CommandResult",
    "RiskLevel",
    "RiskAssessment",
    "assess_risk",
    "is_interactive",
    "PermissionDecision",
    "PermissionConfig",
    "LocalExecutorConfig",
    "ConfigLoader",
    "ConfigWatcher",
    # New features
    "CommandHandle",
    "DoomLoopDetector",
    "generate_json_schema",
    "wildcard_match",
    # Permission helpers
    "BashPermissionRules",
    "ExternalDirectoryRules",
    "PermissionBlock",
    "_extract_command_base",
    # Output helpers
    "_smart_truncate",
    "_save_output_to_file",
    "_check_env_file_exposure",
    "ENV_FILE_PATTERNS",
    # Path containment
    "contains_path",
    "contains_path_lexical",
    "detect_symlink_escape",
    "_resolve_workdir",
    "extract_external_command_paths",
    "_extract_command_tokens",
    # Shell discovery
    "_find_git_bash",
    "_default_shell",
    "_shell_argv",
    # Constants
    "DEFAULT_TIMEOUT_MS",
    "MAX_TIMEOUT_MS",
    "MAX_CAPTURE_BYTES",
    "MAX_OUTPUT_LINES",
    "IS_WINDOWS",
]


# ══════════════════════════════════════════════════════════════════════════════
# § 11  CLI ENTRY POINT + TESTS
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys as _sys

    # --generate-schema CLI
    if "--generate-schema" in _sys.argv:
        out = (
            _sys.argv[_sys.argv.index("--generate-schema") + 1]
            if len(_sys.argv) > _sys.argv.index("--generate-schema") + 1
            else "local_executor.schema.json"
        )
        generate_json_schema(output_path=out)
        print(f"Schema written to {out}")
        _sys.exit(0)

    # --run-tests CLI
    if "--run-tests" in _sys.argv:
        print("Running built-in tests...\n")

        # ── 7.1 Wildcard Matching ─────────────────────────────────────
        print("=== Wildcard Matching ===")
        wt = [
            ("git *", "git push origin main", True),
            ("git *", "github-cli auth", False),
            ("git *", "git", False),
            ("git push*", "git push --force", True),
            ("rm *", "rmdir old_build", False),
            ("ls", "ls", True),
            ("ls", "ls -la", False),
            ("ls*", "ls -la", True),
            ("python3*", "python3 -c 'x=1'", True),
            ("python3 -c *", "python3 -c print(1)", True),
        ]
        for pat, val, expected in wt:
            result = wildcard_match(pat, val)
            status = "PASS" if result == expected else "FAIL"
            print(f"  {status}: wildcard_match({pat!r}, {val!r}) = {result}")
        print()

        # ── 7.2 Config Loading ────────────────────────────────────────
        print("=== Config Loading ===")
        import tempfile, json as _json

        with tempfile.TemporaryDirectory() as tmpdir:
            proj_cfg = Path(tmpdir) / ".local_executor" / "local_executor.json"
            proj_cfg.parent.mkdir()
            proj_cfg.write_text(
                _json.dumps(
                    {"timeout_ms": 30000, "permission": {"bash": {"*": "allow"}}}
                )
            )
            loader = ConfigLoader(workspace_dir=tmpdir)
            cfg = loader.load()
            assert cfg.timeout_ms == 30000, f"timeout_ms={cfg.timeout_ms}"
            assert cfg.permission.bash.evaluate("rm -rf /") == "allow"
            print("  PASS: Project config overrides")
        print()

        # ── 7.3 Permission Evaluation (Last-Rule-Wins) ────────────────
        print("=== Permission Evaluation (Last-Rule-Wins) ===")
        rules = BashPermissionRules.from_dict(
            {
                "*": "ask",
                "git *": "allow",
                "git push*": "deny",
                "git push --force*": "deny",
            }
        )
        ev = [
            ("ls -la", "ask"),
            ("git status", "allow"),
            ("git log --oneline", "allow"),
            ("git push origin main", "deny"),
            ("git push --force", "deny"),
        ]
        for cmd, expected in ev:
            result = rules.evaluate(cmd)
            status = "PASS" if result == expected else "FAIL"
            print(f"  {status}: rules.evaluate({cmd!r}) = {result!r}")
        print()

        # ── 7.4 External Directory Rules ──────────────────────────────
        print("=== External Directory Rules ===")
        ext_rules = ExternalDirectoryRules.from_dict(
            {"*": "ask", "/tmp/*": "allow", "~/projects/*": "allow"}
        )
        er = [
            ("/tmp/build", "allow"),
            ("/etc/passwd", "ask"),
            (os.path.expanduser("~/projects/myapp"), "allow"),
        ]
        for d, expected in er:
            result = ext_rules.evaluate(d)
            status = "PASS" if result == expected else "FAIL"
            print(f"  {status}: ext_rules.evaluate({d!r}) = {result!r}")
        print()

        # ── 7.5 Consecutive Denial Tracking ───────────────────────────
        print("=== Consecutive Denial Tracking ===")
        mgr = PermissionManager(config=LocalExecutorConfig(non_interactive=True))
        mgr._record_denial("rm -rf /")
        mgr._record_denial("rm -rf /")
        result = mgr._check_consecutive_denials("rm -rf /")
        assert result is not None, "Expected auto-deny"
        print("  PASS: Auto-deny after 2 consecutive denials")

        # Threshold=3 test
        mgr2 = PermissionManager(config=LocalExecutorConfig(non_interactive=False))
        mgr2._config = LocalExecutorConfig(consecutive_deny_threshold=3)
        mgr2._record_denial("sudo something")
        mgr2._record_denial("sudo something")
        result2 = mgr2._check_consecutive_denials("sudo something")
        assert result2 is None, "Should NOT auto-deny after only 2 with threshold=3"
        print("  PASS: No auto-deny with threshold=3 after 2 denials")
        mgr2._record_denial("sudo something")
        result3 = mgr2._check_consecutive_denials("sudo something")
        assert result3 is not None, "Should auto-deny after 3 with threshold=3"
        print("  PASS: Auto-deny after 3 with threshold=3")
        print()

        # ── 7.6 BashInput Validation ──────────────────────────────────
        print("=== BashInput Validation ===")
        BashInput(command="ls -la")
        print("  PASS: Normal command accepted")
        try:
            BashInput(command="   ")
            print("  FAIL: Whitespace-only should be rejected")
        except Exception:
            print("  PASS: Whitespace-only rejected")
        try:
            BashInput(command="")
            print("  FAIL: Empty string should be rejected")
        except Exception:
            print("  PASS: Empty string rejected")
        print()

        # ── 7.7 DoomLoopDetector ──────────────────────────────────────
        print("=== DoomLoopDetector ===")
        dd = DoomLoopDetector(threshold=3)
        assert dd.record("echo hi") == False
        assert dd.record("echo hi") == False
        assert dd.record("echo hi") == True
        print("  PASS: Doom loop detected after 3 identical commands")
        dd.reset()
        assert dd.record("echo hi") == False
        print("  PASS: Reset works")
        print()

        # ── 7.8 CommandHandle ─────────────────────────────────────────
        print("=== CommandHandle ===")
        ch = CommandHandle()
        assert not ch._cancel_event.is_set()
        ch.cancel()
        assert ch._cancel_event.is_set()
        print("  PASS: CommandHandle cancel works")
        print()

        # ── 7.9 RiskAssessment ────────────────────────────────────────
        print("=== RiskAssessment ===")
        ra = RiskAssessment(level=RiskLevel.MODERATE, reasons=["git push"])
        assert ra.level == RiskLevel.MODERATE
        assert not ra.is_interactive
        ra2 = RiskAssessment(level=RiskLevel.HIGH, reasons=["interactive shell"])
        assert ra2.is_interactive
        print("  PASS: RiskAssessment model works")
        print()

        # ── 7.10 Env File Protection ──────────────────────────────────
        print("=== Env File Protection ===")
        assert _check_env_file_exposure("cat .env") == True
        assert _check_env_file_exposure("cat .env.local") == True
        assert _check_env_file_exposure("echo hello") == False
        assert _check_env_file_exposure("cat /path/to/.env.production") == True
        print("  PASS: Env file detection works")
        print()

        # ── 7.11 parse_cd_target ──────────────────────────────────────
        print("=== parse_cd_target ===")
        if IS_WINDOWS:
            base = os.getcwd()
            assert parse_cd_target("cd subdir", base) == os.path.join(base, "subdir")
        else:
            assert parse_cd_target("cd /tmp", "/home") == "/tmp"
            assert parse_cd_target("cd subdir", "/home") == os.path.join(
                "/home", "subdir"
            )
        assert parse_cd_target("cd && ls", os.getcwd()) is None
        assert parse_cd_target("ls", os.getcwd()) is None
        print("  PASS: parse_cd_target works")
        print()

        print("All built-in tests passed!")
        _sys.exit(0)
