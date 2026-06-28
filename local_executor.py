"""
local_executor.py
=================
Local code-execution tool harness with HITL permission gating.

Mirrors exactly:
  • OpenCode's bash.ts  — permission system, process runner, timeout/kill chain
                          (SIGTERM → wait → SIGKILL), output truncation,
                          external-directory advisory scan, saved "always" rules
  • ADK's code executor — Python snippet execution, OUTCOME_OK / OUTCOME_ERROR
                          result model that the LLM understands

Drop-in tool functions for ADK, LangChain, raw LiteLLM, or any Python agent.

Requirements
------------
    pip install pydantic>=2.0
    (everything else is stdlib)

Quick Start
-----------
    from local_executor import LocalExecutorToolkit

    toolkit = LocalExecutorToolkit(require_permission=True)
    execute_python_code, execute_shell_command = toolkit.get_tool_functions()

    # ADK:
    root_agent = LlmAgent(name="agent", model=model,
                          tools=[execute_python_code, execute_shell_command])
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TIMEOUT_MS: int = 2 * 60 * 1_000  # 2 min  (same as OpenCode)
MAX_TIMEOUT_MS: int = 10 * 60 * 1_000  # 10 min (same as OpenCode)
MAX_CAPTURE_BYTES: int = 1024 * 1024  # 1 MB   (same as OpenCode)
FORCE_KILL_DELAY_S: float = 3.0  # SIGTERM grace period before SIGKILL

_IS_WINDOWS: bool = platform.system() == "Windows"


def _find_git_bash() -> str | None:
    """Try to find Git Bash on Windows."""
    import shutil

    # Common Git Bash locations
    candidates = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Git\bin\bash.exe"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    # Check if bash is on PATH (e.g., MSYS2, Cygwin)
    bash_on_path = shutil.which("bash")
    if bash_on_path:
        return bash_on_path
    return None


def _default_shell() -> str:
    """
    Platform default shell — mirrors OpenCode's defaultShell().

    On Windows, checks for Git Bash / MSYS2 / Cygwin first (for Unix commands
    like ls, grep, etc.). Falls back to cmd.exe if none found.

    Override with SHELL environment variable.
    """
    # Allow explicit override via environment variable
    env_shell = os.environ.get("SHELL")
    if env_shell:
        return env_shell

    if _IS_WINDOWS:
        # Try Git Bash / MSYS2 / Cygwin first for Unix command support
        git_bash = _find_git_bash()
        if git_bash:
            return git_bash
        return os.environ.get("COMSPEC", "cmd.exe")
    return "/bin/sh"


# asyncio.to_thread arrived in 3.9; polyfill for 3.8
try:
    from asyncio import to_thread as _to_thread  # type: ignore
except ImportError:
    import functools

    async def _to_thread(func, *args, **kwargs):  # type: ignore
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, functools.partial(func, *args, **kwargs)
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────


class Outcome(str, Enum):
    """Mirrors ADK's code_execution_result outcome values."""

    OK = "OUTCOME_OK"
    ERROR = "OUTCOME_ERROR"
    TIMEOUT = "OUTCOME_TIMEOUT"
    DENIED = "OUTCOME_DENIED"  # blocked by permission system


class CodeInput(BaseModel):
    """Input model for the Python code executor."""

    code: str = Field(..., description="Python source code to execute")
    workdir: str = Field(
        ".", description="Working directory. Relative paths resolve from CWD."
    )
    timeout_ms: int = Field(
        DEFAULT_TIMEOUT_MS,
        ge=1,
        le=MAX_TIMEOUT_MS,
        description=f"Timeout in ms. Default {DEFAULT_TIMEOUT_MS}, max {MAX_TIMEOUT_MS}.",
    )


class ShellInput(BaseModel):
    """Input model for the shell command executor."""

    command: str = Field(..., description="Shell command string to execute")
    workdir: str = Field(".", description="Working directory.")
    timeout_ms: int = Field(
        DEFAULT_TIMEOUT_MS,
        ge=1,
        le=MAX_TIMEOUT_MS,
        description=f"Timeout in ms. Default {DEFAULT_TIMEOUT_MS}, max {MAX_TIMEOUT_MS}.",
    )


class ExecutionResult(BaseModel):
    """
    Unified result for code and shell execution.

    The `to_adk_dict()` method returns the ADK-compatible format
    (outcome + output) that the LLM sees.
    """

    outcome: Outcome
    output: str = ""
    exit_code: Optional[int] = None
    truncated: bool = False
    timed_out: bool = False
    warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None

    def to_adk_dict(self) -> dict:
        """
        ADK-style result dict — identical structure to code_execution_result.
        The LLM reads `outcome` and `output` from here.
        """
        parts = [self.output] if self.output else []
        if self.error:
            parts.append(f"Error: {self.error}")
        if self.warnings:
            parts.append("Warnings:\n" + "\n".join(f"  - {w}" for w in self.warnings))
        if self.timed_out:
            parts.append(f"Command timed out.")
        if self.truncated:
            parts.append("[Output truncated at 1 MB safety limit]")
        if self.exit_code is not None and self.exit_code != 0:
            parts.append(f"Exit code: {self.exit_code}")

        return {
            "outcome": self.outcome.value,
            "output": "\n\n".join(parts) if parts else "(no output)",
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: PERMISSION SYSTEM  (mirrors OpenCode's permission.ts)
# ─────────────────────────────────────────────────────────────────────────────


class PermissionEffect(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class PermissionRule(BaseModel):
    """
    A single permission rule with wildcard matching.
    Mirrors OpenCode's Permission.Rule — last-match-wins semantics.
    """

    action: str  # e.g. "python_executor", "bash", "external_directory"
    resource: str  # fnmatch pattern, e.g. "ls *" or "*"
    effect: PermissionEffect


class PermissionConfig(BaseModel):
    """
    Global permission policy applied before checking saved rules.

    Examples
    --------
    # Never ask for Python execution (trust the model):
    PermissionConfig(allowed_actions=["python_executor"])

    # Never allow shell at all:
    PermissionConfig(denied_actions=["bash"])

    # Testing / CI — approve everything silently:
    PermissionConfig(auto_approve=True)
    """

    # Actions always denied without prompting
    denied_actions: list[str] = Field(default_factory=list)
    # Actions always allowed without prompting
    allowed_actions: list[str] = Field(default_factory=list)
    # Skip all prompts and auto-approve (useful for testing/CI)
    auto_approve: bool = False


def _wildcard_match(pattern: str, value: str) -> bool:
    """fnmatch-based wildcard matching (same semantics as OpenCode's Wildcard.match)."""
    return fnmatch.fnmatch(value, pattern)


class PermissionDeniedError(Exception):
    """Raised when a deny rule blocks the action."""

    def __init__(self, action: str, resource: str) -> None:
        super().__init__(f"Denied: {action!r} on {resource!r}")
        self.action, self.resource = action, resource


class PermissionRejectedError(Exception):
    """Raised when the user explicitly rejects a prompt."""

    def __init__(self, action: str, resource: str) -> None:
        super().__init__(f"Rejected by user: {action!r} on {resource!r}")
        self.action, self.resource = action, resource


class PermissionSystem:
    """
    HITL permission gate — exact mirror of OpenCode's PermissionV2.

    Resolution order (last-match wins, same as OpenCode):
      1. Global deny list   → PermissionDeniedError immediately
      2. auto_approve flag  → allow silently
      3. Global allow list  → allow silently
      4. Saved always-rules → allow or deny per saved preference
      5. Default            → prompt the terminal user (Y / Always / N)

    Persistence
    -----------
    Save:    perm.save_rules("rules.json")
    Restore: perm.load_rules("rules.json")
    """

    def __init__(self, config: PermissionConfig | None = None) -> None:
        self._config = config or PermissionConfig()
        self._saved_rules: list[PermissionRule] = []
        self._lock = threading.Lock()

    # ── Sync interface ────────────────────────────────────────────────────────

    def assert_permission(
        self,
        action: str,
        resource: str,
        *,
        hint: str = "",
    ) -> None:
        """
        Ensure the action is permitted.
        Raises PermissionDeniedError or PermissionRejectedError on failure.
        Blocks on terminal prompt if no saved rule matches.
        """
        effect = self._evaluate(action, resource)
        if effect == PermissionEffect.DENY:
            raise PermissionDeniedError(action, resource)
        if effect == PermissionEffect.ALLOW:
            return
        # ASK — prompt the user
        self._terminal_prompt(action, resource, hint=hint)

    # ── Async interface ───────────────────────────────────────────────────────

    async def assert_permission_async(
        self,
        action: str,
        resource: str,
        *,
        hint: str = "",
    ) -> None:
        """
        Async version — runs the blocking input() in a thread so the event loop
        isn't stalled during human approval.
        """
        effect = self._evaluate(action, resource)
        if effect == PermissionEffect.DENY:
            raise PermissionDeniedError(action, resource)
        if effect == PermissionEffect.ALLOW:
            return
        await _to_thread(self._terminal_prompt, action, resource, hint)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_rules(self, path: str) -> None:
        """Persist saved always-rules to a JSON file."""
        with self._lock:
            data = [r.model_dump() for r in self._saved_rules]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def load_rules(self, path: str) -> None:
        """Restore saved always-rules from a JSON file."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        with self._lock:
            self._saved_rules = [PermissionRule(**r) for r in data]

    @property
    def saved_rules(self) -> list[PermissionRule]:
        with self._lock:
            return list(self._saved_rules)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _evaluate(self, action: str, resource: str) -> PermissionEffect:
        """
        Evaluate the effective permission for (action, resource).
        Mirrors OpenCode's evaluateInput() + evaluate() functions.
        """
        # 1. Global deny
        if any(_wildcard_match(p, action) for p in self._config.denied_actions):
            return PermissionEffect.DENY

        # 2. Auto-approve / global allow
        if self._config.auto_approve:
            return PermissionEffect.ALLOW
        if any(_wildcard_match(p, action) for p in self._config.allowed_actions):
            return PermissionEffect.ALLOW

        # 3. Saved rules — last-match wins (OpenCode semantics)
        with self._lock:
            matched: PermissionEffect | None = None
            for rule in self._saved_rules:
                if _wildcard_match(rule.action, action) and _wildcard_match(
                    rule.resource, resource
                ):
                    matched = rule.effect
        if matched is not None:
            return matched

        # 4. Default: ask
        return PermissionEffect.ASK

    def _terminal_prompt(self, action: str, resource: str, *, hint: str) -> None:
        """
        Block and prompt the terminal user.
        Mirrors OpenCode's user-facing permission dialog.
        """
        SEP = "─" * 66
        # Truncate very long resource strings for display
        display_resource = resource if len(resource) <= 200 else resource[:197] + "..."

        print(f"\n{SEP}")
        print("  ⚠   PERMISSION REQUIRED")
        print(SEP)
        print(f"  Action   : {action}")
        print(f"  Resource : {display_resource}")
        if hint:
            print(f"  Detail   :")
            for line in hint.splitlines():
                print(f"    {line}")
        print(SEP)
        print("    [y] Allow this once")
        print("    [a] Always allow  (saves rule — never asked again for this)")
        print("    [n] Deny")
        print(SEP)

        while True:
            try:
                raw = input("  Your choice [y/a/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                raw = "n"

            if raw in ("y", "yes"):
                print(f"  ✓ Allowed (once)\n")
                return

            if raw in ("a", "always"):
                rule = PermissionRule(
                    action=action,
                    resource=resource,
                    effect=PermissionEffect.ALLOW,
                )
                with self._lock:
                    self._saved_rules.append(rule)
                print(f"  ✓ Rule saved — always allow {action!r}\n")
                return

            if raw in ("n", "no", "deny"):
                print(f"  ✗ Denied\n")
                raise PermissionRejectedError(action, resource)

            print("  Please type  y, a, or n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: FS UTILITIES  (mirrors OpenCode's fs-util.ts relevant parts)
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_cwd(workdir: str) -> str:
    """
    Resolve workdir to an absolute, canonical directory path.
    Raises FileNotFoundError / NotADirectoryError on bad input.
    Mirrors OpenCode's LocationMutation.resolve() — basic form.
    """
    p = Path(workdir).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Working directory not found: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"Working directory is not a directory: {p}")
    return str(p)


def _path_contains(parent: str, child: str) -> bool:
    """
    True if `child` is inside `parent`.
    Mirrors OpenCode's FSUtil.contains().
    """
    try:
        Path(child).relative_to(parent)
        return True
    except ValueError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: EXTERNAL DIRECTORY SCANNER  (mirrors OpenCode's advisory scan)
# ─────────────────────────────────────────────────────────────────────────────


def _shell_tokens(command: str) -> list[str]:
    """
    Tokenise a shell command string, respecting single/double quotes.
    Primary: shlex.split(); fallback: OpenCode's regex pattern.
    """
    try:
        return shlex.split(command)
    except ValueError:
        return re.findall(r"""(?:[^\s"']+|"[^"]*"|'[^']*')+""", command) or []


def _external_command_directories(command: str, cwd: str) -> list[str]:
    """
    Advisory scan: find absolute paths in `command` that live outside `cwd`.
    Returns parent directories that are external.

    This is a BEST-EFFORT scan, NOT a security gate.
    Mirrors OpenCode's externalCommandDirectories() exactly.
    """
    cwd_real = os.path.realpath(cwd)
    seen: set[str] = set()

    for token in _shell_tokens(command):
        # Strip surrounding quotes, then trailing shell operators
        value = token.strip("'\"")
        value = re.sub(r"[;,|&]+$", "", value)

        if not os.path.isabs(value):
            continue

        try:
            resolved = os.path.realpath(value)
        except OSError:
            resolved = os.path.normpath(value)

        if _path_contains(cwd_real, resolved):
            continue  # inside cwd — not external

        parent = os.path.dirname(resolved)
        seen.add(parent)

    return sorted(seen)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: PROCESS RUNNER  (mirrors OpenCode's AppProcess + CrossSpawnSpawner)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _StreamResult:
    data: bytes
    truncated: bool


def _read_stream(stream, max_bytes: int, holder: list) -> None:
    """
    Thread worker: drain `stream` into a byte buffer, capped at `max_bytes`.
    Mirrors OpenCode's collectStream() fold logic exactly.
    """
    chunks: list[bytes] = []
    total = 0
    truncated = False

    for chunk in iter(lambda: stream.read(8192), b""):
        remaining = max_bytes - total
        if remaining > 0:
            chunks.append(chunk[:remaining] if len(chunk) > remaining else chunk)
        total += len(chunk)
        if total > max_bytes:
            truncated = True

    holder.append(_StreamResult(data=b"".join(chunks), truncated=truncated))


def _kill_process_tree(proc: subprocess.Popen, force: bool = False) -> None:
    """
    Kill the process (and its group/tree on every platform).
    Mirrors OpenCode's killGroup() for POSIX and Windows.

    POSIX : os.killpg(-pid, signal)  → kills the whole process group
    Windows: taskkill /pid PID /T /F → kills the process tree
    """
    if _IS_WINDOWS:
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
    else:
        sig = signal.SIGKILL if force else signal.SIGTERM
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except ProcessLookupError:
            pass  # already gone
        except Exception:
            try:
                proc.send_signal(sig)
            except Exception:
                pass


class ProcessRunner:
    """
    Core subprocess execution engine.

    Mirrors OpenCode's AppProcess.run() + CrossSpawnSpawner behaviour:
      • stdout and stderr merged into one stream (combineOutput = True)
      • byte-cap truncation at max_bytes
      • timeout → SIGTERM → wait FORCE_KILL_DELAY_S seconds → SIGKILL
      • detached process group on POSIX for clean group kill
      • WindowsHide on Windows
    """

    def run(
        self,
        argv: list[str],
        *,
        cwd: str,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        max_bytes: int = MAX_CAPTURE_BYTES,
        env: dict | None = None,
    ) -> ExecutionResult:
        """
        Execute `argv` in `cwd` and return an ExecutionResult.

        Parameters
        ----------
        argv        : Command + arguments (pre-split list)
        cwd         : Absolute working directory (must exist)
        timeout_ms  : Kill process after this many ms (0 = no timeout)
        max_bytes   : Max stdout+stderr bytes to capture in memory
        env         : Extra environment variables merged into os.environ
        """
        timeout_s: float | None = (timeout_ms / 1000.0) if timeout_ms > 0 else None
        run_env = {**os.environ, **(env or {})}

        spawn_kwargs: dict = {
            "cwd": cwd,
            "env": run_env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,  # merge stderr → stdout (combineOutput=True)
            "stdin": subprocess.DEVNULL,
        }
        if _IS_WINDOWS:
            # Hide the console window that would otherwise pop up
            spawn_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore
        else:
            # Create a new process group so we can kill the whole tree cleanly
            spawn_kwargs["start_new_session"] = True

        try:
            proc = subprocess.Popen(argv, **spawn_kwargs)
        except FileNotFoundError as exc:
            return ExecutionResult(
                outcome=Outcome.ERROR, error=f"Command not found: {exc}"
            )
        except PermissionError as exc:
            return ExecutionResult(
                outcome=Outcome.ERROR,
                error=f"Permission error launching process: {exc}",
            )
        except OSError as exc:
            return ExecutionResult(
                outcome=Outcome.ERROR, error=f"OS error launching process: {exc}"
            )

        # Read stdout (+ merged stderr) in a background thread so we don't deadlock
        out_holder: list[_StreamResult] = []
        reader = threading.Thread(
            target=_read_stream,
            args=(proc.stdout, max_bytes, out_holder),
            daemon=True,
        )
        reader.start()

        timed_out = False
        exit_code: int | None = None

        try:
            proc.wait(timeout=timeout_s)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            # ── Graceful kill first (SIGTERM / taskkill) ──
            _kill_process_tree(proc, force=False)
            try:
                proc.wait(timeout=FORCE_KILL_DELAY_S)
            except subprocess.TimeoutExpired:
                # ── Force kill (SIGKILL) ──
                _kill_process_tree(proc, force=True)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass  # process is as dead as it's going to get

        # Wait for the reader to finish draining whatever the process left in the pipe
        reader.join(timeout=5)

        collected = (
            out_holder[0] if out_holder else _StreamResult(data=b"", truncated=False)
        )
        raw_output = collected.data.decode("utf-8", errors="replace") or "(no output)"

        if timed_out:
            return ExecutionResult(
                outcome=Outcome.TIMEOUT,
                output=raw_output,
                timed_out=True,
                truncated=collected.truncated,
                error=f"Command exceeded timeout of {timeout_ms} ms. "
                "Retry with a larger timeout_ms if needed.",
            )

        return ExecutionResult(
            outcome=Outcome.OK if exit_code == 0 else Outcome.ERROR,
            output=raw_output,
            exit_code=exit_code,
            truncated=collected.truncated,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: PYTHON CODE EXECUTOR TOOL
# ─────────────────────────────────────────────────────────────────────────────


class PythonExecutorTool:
    """
    Execute Python code snippets in an isolated subprocess.

    Mirrors ADK's built-in code executor:
      • Writes code to a temp file → runs `python temp.py` → captures output
      • Returns OUTCOME_OK / OUTCOME_ERROR / OUTCOME_DENIED / OUTCOME_TIMEOUT
      • Output is stdout + stderr merged (same as ADK)

    Usage (sync)
    ------------
        tool   = PythonExecutorTool()
        result = tool.execute(CodeInput(code="print(2 ** 10)"))
        print(result.output)      # "1024"
        print(result.to_adk_dict())  # {"outcome": "OUTCOME_OK", "output": "1024"}

    Usage (async)
    -------------
        result = await tool.execute_async(CodeInput(code="import math; print(math.pi)"))
    """

    TOOL_NAME = "python_executor"
    TOOL_DESCRIPTION = (
        "Execute Python code locally in an isolated subprocess. "
        "Returns stdout+stderr combined as 'output'. "
        f"Default timeout: {DEFAULT_TIMEOUT_MS} ms; max: {MAX_TIMEOUT_MS} ms. "
        f"Output capped at {MAX_CAPTURE_BYTES // 1024} KB. "
        "Always requires user permission before running (unless auto_approve=True)."
    )

    def __init__(
        self,
        *,
        permission_system: PermissionSystem | None = None,
        python_executable: str | None = None,
        max_bytes: int = MAX_CAPTURE_BYTES,
        require_permission: bool = True,
    ) -> None:
        self._perm = permission_system or PermissionSystem()
        self._python = python_executable or sys.executable
        self._runner = ProcessRunner()
        self._max = max_bytes
        self._require = require_permission

    # ── Public sync API ───────────────────────────────────────────────────────

    def execute(self, inp: CodeInput | dict | str) -> ExecutionResult:
        """Execute Python code (sync, blocking)."""
        inp = _coerce_code_input(inp)
        try:
            cwd = _resolve_cwd(inp.workdir)
        except (FileNotFoundError, NotADirectoryError) as exc:
            return ExecutionResult(outcome=Outcome.ERROR, error=str(exc))

        if self._require:
            # Show first 120 chars of the code as the "resource" being requested
            try:
                self._perm.assert_permission(
                    action=self.TOOL_NAME,
                    resource=inp.code[:120].replace("\n", " "),
                    hint=f"Execute Python in: {cwd}",
                )
            except (PermissionDeniedError, PermissionRejectedError) as exc:
                return ExecutionResult(outcome=Outcome.DENIED, error=str(exc))

        return self._run_code(inp, cwd)

    # ── Public async API ──────────────────────────────────────────────────────

    async def execute_async(self, inp: CodeInput | dict | str) -> ExecutionResult:
        """Execute Python code (async — permission prompt + subprocess in threads)."""
        inp = _coerce_code_input(inp)
        try:
            cwd = _resolve_cwd(inp.workdir)
        except (FileNotFoundError, NotADirectoryError) as exc:
            return ExecutionResult(outcome=Outcome.ERROR, error=str(exc))

        if self._require:
            try:
                await self._perm.assert_permission_async(
                    action=self.TOOL_NAME,
                    resource=inp.code[:120].replace("\n", " "),
                    hint=f"Execute Python in: {cwd}",
                )
            except (PermissionDeniedError, PermissionRejectedError) as exc:
                return ExecutionResult(outcome=Outcome.DENIED, error=str(exc))

        # subprocess.Popen + proc.wait() are blocking — offload to thread pool
        return await _to_thread(self._run_code, inp, cwd)

    # ── as_tool_function — drop into any agent's tools= list ─────────────────

    def as_tool_function(self):
        """
        Return a plain callable for ADK / LangChain / raw LiteLLM tool lists.

            tools=[python_tool.as_tool_function()]

        The returned function's docstring is used by the LLM to understand the tool.
        """
        tool = self

        def execute_python_code(
            code: str,
            timeout_ms: int = DEFAULT_TIMEOUT_MS,
        ) -> dict:
            """
            Execute Python code locally in a subprocess and return the output.

            Use this to evaluate mathematical expressions, run data processing,
            perform calculations, or execute any Python snippet.

            Args:
                code:       Complete Python code to execute. Print values you want
                            to see — the tool captures stdout and stderr.
                timeout_ms: Timeout in milliseconds (default 120000, max 600000).
                            Increase for long-running computations.

            Returns:
                dict with keys:
                  outcome : "OUTCOME_OK" | "OUTCOME_ERROR" | "OUTCOME_TIMEOUT" | "OUTCOME_DENIED"
                  output  : captured stdout + stderr as a single string
            """
            result = tool.execute(CodeInput(code=code, timeout_ms=timeout_ms))
            return result.to_adk_dict()

        return execute_python_code

    # ── Internal ─────────────────────────────────────────────────────────────

    def _run_code(self, inp: CodeInput, cwd: str) -> ExecutionResult:
        """Write code to a temp file, execute it, clean up, return result."""
        tmp_path: str | None = None
        try:
            # Write to a temp file in the workdir so relative imports work
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                prefix="_agent_exec_",
                dir=cwd,
                delete=False,
                encoding="utf-8",
            ) as fh:
                fh.write(inp.code)
                tmp_path = fh.name

            return self._runner.run(
                [self._python, tmp_path],
                cwd=cwd,
                timeout_ms=inp.timeout_ms,
                max_bytes=self._max,
            )
        except Exception as exc:
            return ExecutionResult(
                outcome=Outcome.ERROR, error=f"Execution setup failed: {exc}"
            )
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: SHELL EXECUTOR TOOL  (mirrors OpenCode's bash tool exactly)
# ─────────────────────────────────────────────────────────────────────────────


class ShellExecutorTool:
    """
    Execute shell commands on the local machine.

    Exact feature parity with OpenCode's bash.ts:
      ✔ Permission gate (ask / allow-once / always / deny)
      ✔ External directory advisory scan → warnings (does NOT block execution)
      ✔ Timeout with SIGTERM → 3 s → SIGKILL escalation
      ✔ Output truncation at MAX_CAPTURE_BYTES (1 MB)
      ✔ Merged stdout + stderr output
      ✔ Cross-platform shell (/bin/sh on POSIX, cmd.exe on Windows)
      ✔ Detached process group on POSIX for clean kill

    Usage (sync)
    ------------
        tool   = ShellExecutorTool()
        result = tool.execute(ShellInput(command="ls -la"))
        print(result.output)

    Usage (async)
    -------------
        result = await tool.execute_async(ShellInput(command="npm run build"))
    """

    TOOL_NAME = "bash"
    TOOL_DESCRIPTION = (
        "Execute a shell command on the local machine with full filesystem, "
        "process, and network authority. "
        f"Default timeout: {DEFAULT_TIMEOUT_MS} ms; max: {MAX_TIMEOUT_MS} ms. "
        f"Output capped at {MAX_CAPTURE_BYTES // 1024} KB. "
        f"Shell: /bin/sh (POSIX) or cmd.exe (Windows). "
        "Always requires user permission before running."
    )

    def __init__(
        self,
        *,
        permission_system: PermissionSystem | None = None,
        shell: str | None = None,
        max_bytes: int = MAX_CAPTURE_BYTES,
        require_permission: bool = True,
    ) -> None:
        self._perm = permission_system or PermissionSystem()
        self._shell = shell or _default_shell()
        self._runner = ProcessRunner()
        self._max = max_bytes
        self._require = require_permission

    # ── Public sync API ───────────────────────────────────────────────────────

    def execute(self, inp: ShellInput | dict | str) -> ExecutionResult:
        """Execute a shell command (sync, blocking)."""
        inp = _coerce_shell_input(inp)
        try:
            cwd = _resolve_cwd(inp.workdir)
        except (FileNotFoundError, NotADirectoryError) as exc:
            return ExecutionResult(outcome=Outcome.ERROR, error=str(exc))

        # Advisory external-directory scan (warnings only, never blocks)
        warnings = _build_advisory_warnings(inp.command, cwd)

        if self._require:
            try:
                self._perm.assert_permission(
                    action=self.TOOL_NAME,
                    resource=inp.command,
                    hint=f"Run in: {cwd}"
                    + (
                        f"\nAdvisory warnings:\n  " + "\n  ".join(warnings)
                        if warnings
                        else ""
                    ),
                )
            except (PermissionDeniedError, PermissionRejectedError) as exc:
                return ExecutionResult(
                    outcome=Outcome.DENIED,
                    error=str(exc),
                    warnings=warnings,
                )

        result = self._run_command(inp, cwd)
        result.warnings.extend(warnings)
        return result

    # ── Public async API ──────────────────────────────────────────────────────

    async def execute_async(self, inp: ShellInput | dict | str) -> ExecutionResult:
        """Execute a shell command (async)."""
        inp = _coerce_shell_input(inp)
        try:
            cwd = _resolve_cwd(inp.workdir)
        except (FileNotFoundError, NotADirectoryError) as exc:
            return ExecutionResult(outcome=Outcome.ERROR, error=str(exc))

        warnings = _build_advisory_warnings(inp.command, cwd)

        if self._require:
            try:
                await self._perm.assert_permission_async(
                    action=self.TOOL_NAME,
                    resource=inp.command,
                    hint=f"Run in: {cwd}",
                )
            except (PermissionDeniedError, PermissionRejectedError) as exc:
                return ExecutionResult(
                    outcome=Outcome.DENIED,
                    error=str(exc),
                    warnings=warnings,
                )

        result = await _to_thread(self._run_command, inp, cwd)
        result.warnings.extend(warnings)
        return result

    # ── as_tool_function — drop into any agent's tools= list ─────────────────

    def as_tool_function(self):
        """Return a plain callable for ADK / LangChain tool lists."""
        tool = self

        def execute_shell_command(
            command: str,
            timeout_ms: int = DEFAULT_TIMEOUT_MS,
        ) -> dict:
            """
            Execute a shell command on the local machine.

            Has full access to the filesystem, running processes, and network.
            stdout and stderr are merged into a single 'output' string.

            Args:
                command:    Shell command string to execute (e.g. "ls -la", "npm test").
                timeout_ms: Timeout in milliseconds (default 120000, max 600000).

            Returns:
                dict with keys:
                  outcome   : "OUTCOME_OK" | "OUTCOME_ERROR" | "OUTCOME_TIMEOUT" | "OUTCOME_DENIED"
                  output    : combined stdout + stderr
                  exit_code : integer exit code (None on timeout)
                  truncated : True if output was cut off at the 1 MB limit
                  warnings  : list of advisory messages about external directory access
            """
            result = tool.execute(ShellInput(command=command, timeout_ms=timeout_ms))
            out = result.to_adk_dict()
            out.update(
                {
                    "exit_code": result.exit_code,
                    "truncated": result.truncated,
                    "warnings": result.warnings,
                }
            )
            return out

        return execute_shell_command

    # ── Internal ─────────────────────────────────────────────────────────────

    def _run_command(self, inp: ShellInput, cwd: str) -> ExecutionResult:
        """Build the argv for the platform shell and delegate to ProcessRunner."""
        shell_name = os.path.basename(self._shell).lower()

        # cmd.exe uses /c, bash/zsh/sh use -c
        if shell_name in ("cmd.exe", "cmd"):
            argv = [self._shell, "/c", inp.command]
        else:
            argv = [self._shell, "-c", inp.command]

        return self._runner.run(
            argv,
            cwd=cwd,
            timeout_ms=inp.timeout_ms,
            max_bytes=self._max,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: TOOLKIT — public API, bundles both tools
# ─────────────────────────────────────────────────────────────────────────────


class LocalExecutorToolkit:
    """
    Convenience bundle: shared permission system + Python tool + Shell tool.

    Quick Start
    -----------
        from local_executor import LocalExecutorToolkit, PermissionConfig

        # Default — asks permission at the terminal before every execution
        toolkit = LocalExecutorToolkit()

        # CI / trusted environment — never ask
        toolkit = LocalExecutorToolkit(
            permission_config=PermissionConfig(auto_approve=True)
        )

        # Allow Python silently, always ask for shell
        toolkit = LocalExecutorToolkit(
            permission_config=PermissionConfig(allowed_actions=["python_executor"])
        )

        # Plug into ADK:
        tools = toolkit.get_tool_functions()
        root_agent = LlmAgent(name="agent", model=model, tools=tools)

    Persistence
    -----------
        toolkit.permissions.save_rules("perm_rules.json")
        toolkit.permissions.load_rules("perm_rules.json")
    """

    def __init__(
        self,
        *,
        permission_config: PermissionConfig | None = None,
        python_executable: str | None = None,
        shell: str | None = None,
        max_bytes: int = MAX_CAPTURE_BYTES,
        require_permission: bool = True,
    ) -> None:
        self.permissions = PermissionSystem(permission_config)

        self.python_tool = PythonExecutorTool(
            permission_system=self.permissions,
            python_executable=python_executable,
            max_bytes=max_bytes,
            require_permission=require_permission,
        )
        self.shell_tool = ShellExecutorTool(
            permission_system=self.permissions,
            shell=shell,
            max_bytes=max_bytes,
            require_permission=require_permission,
        )

    def get_tool_functions(self) -> list:
        """
        Return [execute_python_code, execute_shell_command] as plain callables.
        Pass this list directly to any agent framework's tools= parameter.
        """
        return [
            self.python_tool.as_tool_function(),
            self.shell_tool.as_tool_function(),
        ]

    def get_python_tool_function(self):
        """Return only the Python executor tool function."""
        return self.python_tool.as_tool_function()

    def get_shell_tool_function(self):
        """Return only the shell executor tool function."""
        return self.shell_tool.as_tool_function()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: HELPERS (module-level, reused internally)
# ─────────────────────────────────────────────────────────────────────────────


def _coerce_code_input(inp: CodeInput | dict | str) -> CodeInput:
    if isinstance(inp, str):
        return CodeInput(code=inp)
    if isinstance(inp, dict):
        return CodeInput(**inp)
    return inp


def _coerce_shell_input(inp: ShellInput | dict | str) -> ShellInput:
    if isinstance(inp, str):
        return ShellInput(command=inp)
    if isinstance(inp, dict):
        return ShellInput(**inp)
    return inp


def _build_advisory_warnings(command: str, cwd: str) -> list[str]:
    """Build advisory warning strings for external directory references."""
    return [
        f"Command argument references external directory {d}/*. "
        "Shell runs with host-user filesystem, process, and network authority; "
        "this scan is advisory only."
        for d in _external_command_directories(command, cwd)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Toolkit (recommended entry point)
    "LocalExecutorToolkit",
    # Individual tools
    "PythonExecutorTool",
    "ShellExecutorTool",
    # Permission system
    "PermissionSystem",
    "PermissionConfig",
    "PermissionRule",
    "PermissionEffect",
    "PermissionDeniedError",
    "PermissionRejectedError",
    # Models
    "CodeInput",
    "ShellInput",
    "ExecutionResult",
    "Outcome",
    # Process runner (for custom use)
    "ProcessRunner",
    # Constants
    "DEFAULT_TIMEOUT_MS",
    "MAX_TIMEOUT_MS",
    "MAX_CAPTURE_BYTES",
]
