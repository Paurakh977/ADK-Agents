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
    Each risky command prompts the user:
      [1] Allow this time
      [2] Allow always for this session (exact command)
      [3] Allow command (e.g. "rm") for entire session
      [4] Deny
    Consecutive-denial detection: if the same exact command (or same base
    command) is denied twice, the 3rd attempt is auto-denied with a clear
    message. History is cleared after auto-deny so the 4th attempt prompts again.

  • Platform-aware shell selection
      Windows  →  Git Bash / MSYS2 / Cygwin (validated, with cmd.exe fallback)
      POSIX    →  SHELL  / /bin/sh
    Shell discovery is logged; each candidate is validated before use.
    Runtime fallback: if the primary shell fails to spawn, retries with cmd.exe.

  • Timeout with SIGTERM → SIGKILL escalation (3 s grace period)

  • Output capture with 512 KB safety cap + smart head/tail truncation
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
import logging
import tempfile
import platform
import subprocess
import threading
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Set, Tuple, Callable, Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ══════════════════════════════════════════════════════════════════════════════
# § 1  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_TIMEOUT_MS: int = 2 * 60 * 1_000  # 2 min  — matches OpenCode default
MAX_TIMEOUT_MS: int = 10 * 60 * 1_000  # 10 min — hard ceiling
MAX_CAPTURE_BYTES: int = 512 * 1024  # 512 KB — hard safety cap for output capture
MAX_OUTPUT_LINES: int = 300  # default max lines returned to LLM
HEAD_LINES: int = 50  # lines from start shown on success
TAIL_LINES: int = 250  # lines from end shown on success
TAIL_LINES_ERROR: int = 280  # lines from end shown on failure (errors at bottom)
HEAD_LINES_ERROR: int = 20  # lines from start shown on failure
FORCE_KILL_DELAY_S: float = 3.0  # SIGTERM → SIGKILL grace period
OUTPUT_DIR_NAME: str = "local_executor_outputs"  # temp dir under system temp
OUTPUT_MAX_AGE_HOURS: int = 1  # auto-cleanup files older than this

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


RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
PURPLE = "\033[35m"
CYAN = "\033[36m"
DIM = "\033[2m"
SEP = "─" * 66


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
    (r"\bgit\b", RiskLevel.MODERATE, "git command"),
    (r"\bnpm\s+install\b|\bnpm\s+i\b", RiskLevel.MODERATE, "npm install"),
    (r"\bpip\s+install\b|\bpip3\s+install\b", RiskLevel.MODERATE, "pip install"),
    (
        r"\bapt(-get)?\s+(install|remove|purge)\b",
        RiskLevel.MODERATE,
        "apt package manager",
    ),
    (r"\bbrew\s+install\b", RiskLevel.MODERATE, "Homebrew install"),
    (r"\byarn\s+add\b", RiskLevel.MODERATE, "yarn add"),
]

# Commands that block waiting for stdin — they'd hang until timeout.
INTERACTIVE_PATTERNS: List[Tuple[str, str]] = [
    (r"^python3?\s*$", "interactive Python REPL"),
    (r"^(i?python3?|bpython)\s*$", "interactive Python REPL"),
    (r"^(ba|z|fi|da|k)?sh\s*$", "interactive shell"),
    (r"^ssh\s", "SSH session (interactive)"),
    (r"\bvim?\b|\bnano\b|\bemacs\b", "terminal text editor"),
    (r"\bless\b|\bmore\b|\bman\b", "terminal pager"),
    (r"\btop\b|\bhtop\b|\bbtop\b|\bglances\b", "interactive system monitor"),
    (r"\bmysql\b|\bpsql\b|\bsqlite3\b", "interactive database REPL"),
    (r"\bnode\s*$|\bts-node\s*$", "interactive Node.js REPL"),
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


def _get_output_dir() -> Path:
    """Get or create the temp output directory for saving full command output."""
    d = Path(tempfile.gettempdir()) / OUTPUT_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_output_to_file(output: str, command: str) -> Optional[str]:
    """
    Save full command output to a temp file.

    Returns the file path string, or None on error.
    The agent can later read this file if it needs the full untruncated output.
    """
    try:
        cmd_hash = hashlib.sha1(command.encode()).hexdigest()[:6]
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{cmd_hash}.log"
        path = _get_output_dir() / filename
        path.write_text(output, encoding="utf-8", errors="replace")
        return str(path)
    except OSError:
        return None


def _smart_truncate(output: str, exit_code: Optional[int]) -> Tuple[str, bool, int]:
    """
    Apply head+tail truncation to output.

    Returns (truncated_text, was_truncated, total_lines).
    - On success (exit_code == 0): HEAD_LINES from start + TAIL_LINES from end
    - On failure (exit_code != 0): HEAD_LINES_ERROR from start + TAIL_LINES_ERROR from end
    """
    lines = output.splitlines(keepends=True)
    total = len(lines)

    if total <= MAX_OUTPUT_LINES:
        return output, False, total

    if exit_code is not None and exit_code != 0:
        head_n, tail_n = HEAD_LINES_ERROR, TAIL_LINES_ERROR
    else:
        head_n, tail_n = HEAD_LINES, TAIL_LINES

    head = lines[:head_n]
    tail = lines[-tail_n:]
    omitted = total - head_n - tail_n

    truncated = "".join(head)
    truncated += f"\n... ({omitted} lines omitted) ...\n\n"
    truncated += "".join(tail)
    return truncated, True, total


def _cleanup_old_outputs() -> None:
    """Delete output files older than OUTPUT_MAX_AGE_HOURS."""
    try:
        cutoff = time.time() - (OUTPUT_MAX_AGE_HOURS * 3600)
        for f in _get_output_dir().iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
    except OSError:
        pass


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


class BashInput(BaseModel):
    """
    Input schema for the bash tool.

    Matches OpenCode's Input:
      command : string  — shell command to execute
      workdir : string? — working directory (defaults to workspace root)
      timeout : int?    — timeout in ms (default 120000, max 600000)
    """

    command: str = Field(
        ...,
        description="Shell command string to execute. Supports full shell syntax: "
        "pipes, redirects, &&, ||, subshells, etc.",
        min_length=1,
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

    @field_validator("command")
    @classmethod
    def _command_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("command must not be empty or whitespace-only")
        return v


# ── StructuredOutput (compact metadata — what the model sees) ───────────────


class BashStructuredOutput(BaseModel):
    """
    Compact metadata returned alongside the output text.

    Matches OpenCode's StructuredOutput:
      exit     : number?  — exit code (undefined on timeout)
      truncated: boolean  — was output truncated at 1MB?
      timeout  : boolean? — did it time out?
    """

    exit: Optional[int] = Field(
        default=None,
        description="Process exit code. None if the command timed out.",
    )
    truncated: bool = Field(
        default=False,
        description="True if output was truncated at the 1MB safety limit.",
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
        description="Path to temp file containing the full untruncated output.",
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
        description="True if stderr exceeded the 1MB safety limit (when combine_output=False).",
    )

    def to_model_output(self) -> List[dict]:
        """
        Convert to OpenCode's toModelOutput format.

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
        raw = self.output or "(no output)"
        truncated_text, was_truncated, total_lines = _smart_truncate(raw, self.exit)
        if was_truncated:
            if self.exit is not None and self.exit != 0:
                head_n, tail_n = HEAD_LINES_ERROR, TAIL_LINES_ERROR
            else:
                head_n, tail_n = HEAD_LINES, TAIL_LINES
            truncated_text += (
                f"\n[truncated: {total_lines} total lines, "
                f"showing first {head_n} + last {tail_n}]"
            )
        if self.output_file:
            truncated_text += f"\n[full output saved to: {self.output_file}]"
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
        return "Command completed."

    def to_adk_dict(self) -> dict:
        """
        ADK-compatible dict format.

        Returns the structured fields as a flat dict for agent frameworks
        that expect a single dict response. Output is smart-truncated
        (head+tail) to keep the LLM context manageable.
        """
        truncated_output, was_truncated, total_lines = _smart_truncate(
            self.output or "(no output)", self.exit
        )
        if was_truncated:
            if self.exit is not None and self.exit != 0:
                head_n, tail_n = HEAD_LINES_ERROR, TAIL_LINES_ERROR
            else:
                head_n, tail_n = HEAD_LINES, TAIL_LINES
            truncated_output += (
                f"\n[truncated: {total_lines} total lines, "
                f"showing first {head_n} + last {tail_n}]"
            )
        if self.output_file:
            truncated_output += f"\n[full output saved to: {self.output_file}]"

        return {
            "outcome": self._outcome().value,
            "output": truncated_output,
            "exit_code": self.exit,
            "truncated": was_truncated,
            "timed_out": self.timeout or False,
            "warnings": self.warnings,
            "summary": self._summary_line(),
            "output_file": self.output_file,
            "stdout": self.stdout,
            "stderr": self.stderr,
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


class PermissionConfig(BaseModel):
    """
    Permission policy configuration.

    Matches OpenCode's permission evaluation order:
      1. Global deny list   → PermissionDeniedError immediately
      2. auto_approve flag  → allow silently
      3. Global allow list  → allow silently
      4. Saved always-rules → allow or deny per saved preference
      5. Default            → prompt the terminal user

    Simplified for the local harness:
      auto_approve    — skip all prompts (testing/CI)
      min_ask_level   — commands below this risk level are auto-allowed
      deny_critical   — CRITICAL commands auto-denied without asking
      allowed_actions — actions always allowed (e.g. ["python_executor"])
      denied_actions  — actions always denied (e.g. ["bash"])
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
            output=self.output,
            warnings=self.warnings,
            output_file=self.output_file,
            stdout=self.stdout,
            stderr=self.stderr,
            stderr_truncated=self.stderr_truncated,
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


class PermissionManager:
    """
    Three-outcome permission gate, mirroring OpenCode's PermissionV2:

        ALLOW_ONCE    — user said yes just for this invocation
        ALLOW_SESSION — user said yes; rule saved for the entire session
        DENY          — user said no; execution blocked
    """

    def __init__(self, config: PermissionConfig):
        self.config = config
        # Exact commands allowed for the session (case-sensitive).
        # We also store "prefix*" patterns for wildcard session allows.
        self._session_allows: Set[str] = set()
        # External directories allowed for the session (canonical paths).
        # Checked before prompting in request_external_directory().
        self._session_ext_allows: Set[str] = set()
        # Track recent denials for consecutive-denial detection.
        # Each entry is (exact_command, base_command).
        self._denial_history: List[Tuple[str, str]] = []
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def request(
        self, command: str, risk: RiskLevel, reasons: List[str]
    ) -> PermissionDecision:
        """Gate the command. Returns the decision."""

        # 1. Auto-approve (testing / CI mode)
        if self.config.auto_approve:
            return PermissionDecision.ALLOW_ONCE

        # 2. Safe → always allow
        if risk == RiskLevel.SAFE:
            return PermissionDecision.ALLOW_ONCE

        # 3. Session-cached allow (exact match or base-command wildcard)
        if self._is_session_allowed(command):
            print(f"  {DIM}[permission] session-allowed: {command[:60]}{RESET}")
            return PermissionDecision.ALLOW_ONCE

        # 4. Auto-deny critical if configured
        if self.config.deny_critical and risk == RiskLevel.CRITICAL:
            print(
                f"\n{PURPLE}[PERMISSION] Critical command auto-denied (deny_critical=True).{RESET}"
            )
            return PermissionDecision.DENY

        # 5. Below the ask threshold → allow silently
        if risk.value < self.config.min_ask_level.value:
            return PermissionDecision.ALLOW_ONCE

        # 6. Consecutive-denial detection: same exact cmd denied 2x → auto-deny 3rd
        #    OR same base cmd denied 2x → auto-deny 3rd.
        #    After auto-deny, history is cleared so the 4th attempt prompts again.
        auto_deny_msg = self._check_consecutive_denials(command)
        if auto_deny_msg:
            print(f"\n  {RED}✗ {auto_deny_msg}{RESET}\n")
            return PermissionDecision.DENY

        # 7. Interactive prompt
        return self._prompt(command, risk, reasons)

    def request_external_directory(
        self, directory: str, context: str = "command"
    ) -> PermissionDecision:
        """
        Prompt the user for permission to access a directory outside the workspace.

        Mirrors OpenCode's externalDirectoryPermission flow:
          - Checks session-cache first (no re-prompt for approved dirs)
          - Shows the directory path and context
          - 3 options: allow once, allow for session, deny
          - Session-cached dirs are stored in _session_ext_allows
        """
        if self.config.auto_approve:
            return PermissionDecision.ALLOW_ONCE

        # Check session cache — if this directory was already approved, skip prompt
        canonical = str(Path(directory).resolve())
        with self._lock:
            if canonical in self._session_ext_allows:
                print(
                    f"  {DIM}[permission] session-allowed external dir: {directory[:60]}{RESET}"
                )
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
        print(
            f"  {BOLD}[2]{RESET}  Allow for session (all commands referencing this dir)"
        )
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
            self._session_allows.add(command)

    # ── Denial message for external callers (execute_shell_command) ────────────

    def denial_message(self, command: str) -> str:
        """
        Return a human-readable denial message for the given command.
        Used by execute_shell_command to craft a clear response for the LLM.
        """
        with self._lock:
            history = list(self._denial_history)
        base = _extract_command_base(command)

        # Check if this exact command was recently denied
        exact_count = sum(1 for d in history if d[0] == command)
        base_count = sum(1 for d in history if d[1] == base)

        if exact_count >= 2:
            return (
                f"Permission denied — the user has already refused this exact "
                f"command ({exact_count} times). Something may be wrong with "
                f"this command. Ask the user what they want to do instead."
            )
        if base_count >= 2:
            return (
                f"Permission denied — the user has refused '{base}' commands "
                f"recently ({base_count} times). The user may not want to run "
                f"this type of command. Try a different approach."
            )
        return "Permission denied by user."

    # ── Internal ──────────────────────────────────────────────────────────────

    def _is_session_allowed(self, command: str) -> bool:
        """Check if command matches any session-wide allow (exact or wildcard)."""
        with self._lock:
            if command in self._session_allows:
                return True
            for pattern in self._session_allows:
                if pattern.endswith("*") and command.startswith(pattern[:-1]):
                    return True
        return False

    def _check_consecutive_denials(self, command: str) -> Optional[str]:
        """
        If the last 2 denials were the same exact command, or the same base
        command, auto-deny this attempt with an informative message.

        Returns the denial message string, or None if no auto-deny.
        After auto-deny, clears history so the next attempt prompts again.
        """
        with self._lock:
            history = list(self._denial_history)

        if len(history) < 2:
            return None

        last_two = history[-2:]
        base = _extract_command_base(command)

        # Same exact command denied twice in a row
        if all(d[0] == command for d in last_two):
            with self._lock:
                self._denial_history.clear()
            return (
                f"Auto-denied: the user has already refused this exact command "
                f"twice. Something may be wrong with this command, or the user "
                f"does not want to run it. Ask the user or try a different approach."
            )

        # Same base command denied twice in a row (different exact commands)
        if all(d[1] == base for d in last_two):
            with self._lock:
                self._denial_history.clear()
            return (
                f"Auto-denied: the user has refused '{base}' commands twice "
                f"recently. The user may not want to use this type of command. "
                f"Ask the user or try a completely different approach."
            )

        return None

    def _record_denial(self, command: str) -> None:
        """Record a denied command in the history for consecutive-denial tracking."""
        base = _extract_command_base(command)
        with self._lock:
            self._denial_history.append((command, base))
            # Keep only last 5 to prevent unbounded growth
            if len(self._denial_history) > 5:
                self._denial_history = self._denial_history[-5:]

    def _prompt(
        self, command: str, risk: RiskLevel, reasons: List[str]
    ) -> PermissionDecision:
        """Blocking terminal prompt with 4 options."""
        reason_str = ", ".join(reasons) if reasons else "unknown risk"
        truncated_cmd = command if len(command) <= 72 else command[:69] + "..."
        base = _extract_command_base(command)

        print(f"\n{SEP}")
        print(f"  {BOLD}PERMISSION REQUEST — {risk.label()}{RESET}")
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
        print(SEP)

        while True:
            try:
                choice = input("  Your choice [1/2/3/4]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {RED}✗ Denied (interrupted){RESET}\n{SEP}\n")
                self._record_denial(command)
                return PermissionDecision.DENY

            if choice == "1":
                print(f"  {GREEN}✓ Allowed (once){RESET}\n{SEP}\n")
                return PermissionDecision.ALLOW_ONCE
            elif choice == "2":
                print(f"  {GREEN}✓ Allowed (session — exact command){RESET}\n{SEP}\n")
                with self._lock:
                    self._session_allows.add(command)
                return PermissionDecision.ALLOW_SESSION
            elif choice == "3":
                wildcard = f"{base}*"
                print(
                    f'  {GREEN}✓ Allowed (session — all "{base}" commands){RESET}\n{SEP}\n'
                )
                with self._lock:
                    self._session_allows.add(wildcard)
                return PermissionDecision.ALLOW_SESSION
            elif choice == "4":
                print(f"  {RED}✗ Denied{RESET}\n{SEP}\n")
                self._record_denial(command)
                return PermissionDecision.DENY
            else:
                print("  Please enter 1, 2, 3, or 4.")


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
    try:
        tokens = shlex.split(command)
    except ValueError:
        # Fallback for complex shell syntax shlex can't parse
        tokens = re.findall(r'(?:[^\s"\']+|"[^"]*"|\'[^\']*\')+', command)

    cwd_resolved = str(Path(cwd).resolve())
    external: Set[str] = set()

    for token in tokens:
        # Strip surrounding quotes
        token = re.sub(r"^(['\"])(.*)\1$", r"\2", token)
        # Strip trailing shell operators
        token = re.sub(r"[;,|&]+$", "", token)
        # Normalize Git Bash paths (/c/...) to Windows paths (C:/...)
        token = _normalize_bash_path(token)

        if not os.path.isabs(token):
            continue
        try:
            resolved = str(Path(token).resolve())
            # If it's inside cwd, not external
            if resolved.startswith(cwd_resolved + os.sep) or resolved == cwd_resolved:
                continue
            parent = str(Path(token).parent.resolve())
            external.add(parent)
        except (OSError, ValueError):
            pass

    return sorted(external)


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
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = re.findall(r'(?:[^\s"\']+|"[^"]*"|\'[^\']*\')+', command)

    cwd_resolved = str(Path(cwd).resolve())
    external: List[str] = []

    for token in tokens:
        token = re.sub(r"^(['\"])(.*)\1$", r"\2", token)
        token = re.sub(r"[;,|&]+$", "", token)
        # Normalize Git Bash paths (/c/...) to Windows paths (C:/...)
        token = _normalize_bash_path(token)
        if not os.path.isabs(token):
            continue
        try:
            resolved = str(Path(token).resolve())
            if not contains_path(cwd_resolved, resolved):
                external.append(resolved)
        except (OSError, ValueError):
            pass

    return external


def _extract_command_tokens(command: str) -> List[Tuple[str, str]]:
    """
    Parse command and return (original_token, resolved_path) for each argument.

    Unlike extract_external_command_paths which only returns resolved paths,
    this preserves the original token so callers can check for symlink escapes
    on the unresolved path.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = re.findall(r'(?:[^\s"\']+|"[^"]*"|\'[^\']*\')+', command)

    result: List[Tuple[str, str]] = []
    for token in tokens:
        token = re.sub(r"^(['\"])(.*)\1$", r"\2", token)
        token = re.sub(r"[;,|&]+$", "", token)
        token = _normalize_bash_path(token)
        try:
            resolved = str(Path(token).resolve())
            result.append((token, resolved))
        except (OSError, ValueError):
            pass
    return result


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
    • stdin closed (DEVNULL) — non-interactive only
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
    ) -> CommandResult:
        """Execute command and return a structured CommandResult.

        Args:
            command: Shell command string.
            cwd: Working directory.
            timeout_ms: Timeout in milliseconds.
            combine_output: If True (default), merge stdout+stderr into a single
                           output string. If False, capture them separately and
                           populate the stdout/stderr fields on CommandResult.
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
        warnings = [
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
                    stdin=subprocess.DEVNULL,
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
                    stdin=subprocess.DEVNULL,
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
                        stdin=subprocess.DEVNULL,
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
                        warnings=warnings,
                    )
            else:
                return CommandResult(
                    command=command,
                    exit_code=1,
                    timed_out=False,
                    truncated=False,
                    cwd=cwd,
                    output=f"Failed to start process: {exc}",
                    warnings=warnings,
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

        def _reader() -> None:
            nonlocal total_in, truncated
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read(16_384)
                if not chunk:
                    break
                remaining = MAX_CAPTURE_BYTES - total_in
                if remaining > 0:
                    keep = chunk if len(chunk) <= remaining else chunk[:remaining]
                    chunks.append(keep)
                else:
                    truncated = True
                    # keep draining the pipe so the process doesn't block
                total_in += len(chunk)
                if total_in > MAX_CAPTURE_BYTES:
                    truncated = True
            read_done.set()

        def _stderr_reader() -> None:
            """Read stderr when combine_output=False, with byte limit."""
            nonlocal stderr_total_in, stderr_truncated
            assert proc.stderr is not None
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
            stderr_read_done.set()

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        stderr_reader: Optional[threading.Thread] = None
        if not combine_output and proc.stderr is not None:
            stderr_reader = threading.Thread(target=_stderr_reader, daemon=True)
            stderr_reader.start()

        # Wait for process with timeout
        timed_out = False
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill(proc)

        # Wait for readers to flush (max 5 s after process exits)
        read_done.wait(timeout=5.0)
        reader.join(timeout=1.0)
        stderr_read_done.wait(timeout=5.0)
        if stderr_reader is not None:
            stderr_reader.join(timeout=1.0)

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

        # Save full output to temp file only when output would be truncated
        line_count = len(output.splitlines())
        output_file = (
            _save_output_to_file(output, command)
            if line_count > MAX_OUTPUT_LINES
            else None
        )

        return CommandResult(
            command=command,
            exit_code=exit_code,
            output=output,
            truncated=truncated,
            timed_out=timed_out,
            cwd=cwd,
            warnings=warnings,
            output_file=output_file,
            stdout=stdout_str if not combine_output else None,
            stderr=stderr_str if not combine_output else None,
            stderr_truncated=stderr_truncated if not combine_output else False,
        )

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

    target = m.group(1).strip().strip("\"'")

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
    ):
        self.workspace_dir = str(Path(workspace_dir or os.getcwd()).resolve())
        self._runner = ProcessRunner(shell=shell)
        self._permissions = PermissionManager(permission_config or PermissionConfig())
        self._cwd = self.workspace_dir

    # ── Primary tool: execute_shell_command ───────────────────────────────────

    def get_tool_function(self) -> Callable:
        """Return the primary execute_shell_command tool function."""
        runner = self._runner
        permissions = self._permissions
        cwd_state = [self.workspace_dir]  # mutable single-element list
        call_count = [0]  # mutable counter for throttled cleanup

        def execute_shell_command(
            command: str,
            workdir: Optional[str] = None,
            timeout_ms: int = DEFAULT_TIMEOUT_MS,
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
            raw_workdir = workdir or cwd_state[0]
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
                workdir, self.workspace_dir, cwd_state[0]
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
            call_count[0] += 1
            if call_count[0] % 50 == 0:
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
            cmd_tokens = _extract_command_tokens(command)
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
                if len(checked_ext_dirs) > 3:
                    break  # prompt for at most 3 distinct external dirs

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

            # ── Execute ────────────────────────────────────────────────────
            result = runner.run(command, resolved_cwd, timeout_ms)

            # ── Track cwd if this was a bare `cd` ─────────────────────────
            if result.exit_code == 0:
                new_cwd = parse_cd_target(command, resolved_cwd)
                if new_cwd and Path(new_cwd).is_dir():
                    # Only allow cd within the workspace
                    if contains_path(self.workspace_dir, new_cwd):
                        cwd_state[0] = new_cwd
                    else:
                        # cd outside workspace — warn but still track
                        print(
                            f"  {YELLOW}⚠ Warning: cd to '{new_cwd}' is outside "
                            f"workspace. Future commands will still default to "
                            f"workspace root.{RESET}"
                        )

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
    "PermissionDecision",
    "PermissionConfig",
    # Permission helpers
    "_extract_command_base",
    # Output helpers
    "_smart_truncate",
    "_save_output_to_file",
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
