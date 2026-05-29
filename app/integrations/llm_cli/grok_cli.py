"""xAI Grok Build CLI adapter (``grok -p``, non-interactive / headless mode).

Grok Build is xAI's terminal-native agentic coding tool (binary: ``grok``). OpenSRE
uses it purely as a one-shot text responder inside the ReAct loop, so invocations
run in headless ``-p`` mode with ``--output-format plain`` and ``--no-auto-update``
and never pass ``--always-approve`` (we do not want Grok autonomously editing files
or running shell commands; OpenSRE provides its own tools).

Env vars
--------
GROK_CLI_BIN              Optional explicit path to the ``grok`` binary.
                          Blank or non-runnable paths are ignored; PATH + fallbacks apply.
GROK_CLI_MODEL            Optional model override (e.g. ``grok-build-0.1``).
                          Unset or empty → omit ``-m``; the CLI's configured default applies.
GROK_CLI_TIMEOUT_SECONDS  Optional invocation timeout override in seconds for long prompts
                          (default: 300, min: 30, max: 600).
XAI_API_KEY               API-key auth for headless/CI runs. Forwarded explicitly to the
                          Grok subprocess via ``CLIInvocation.env`` (see Auth below).

Auth
----
Grok resolves credentials in the order ``model.api_key > model.env_key > active
session token > XAI_API_KEY`` (https://docs.x.ai/build/cli/headless-scripting).
``XAI_API_KEY`` is a secret, so it is forwarded **only** to the Grok subprocess via
``CLIInvocation.env`` rather than the blanket ``_SAFE_SUBPROCESS_ENV_PREFIXES``
allowlist (which would leak it into every other CLI subprocess — same rationale as
the Copilot/Claude Code adapters). There is no documented ``grok auth status``
command, so detection is best-effort: ``XAI_API_KEY`` → authenticated; otherwise a
non-empty session credential file under ``~/.grok-build`` / ``~/.grok`` →
authenticated; otherwise not logged in. The runner re-verifies at invoke time.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.integrations.llm_cli.base import CLIInvocation, CLIProbe
from app.integrations.llm_cli.binary_resolver import (
    candidate_binary_names as _candidate_binary_names,
)
from app.integrations.llm_cli.binary_resolver import (
    default_cli_fallback_paths as _default_cli_fallback_paths,
)
from app.integrations.llm_cli.binary_resolver import (
    resolve_cli_binary,
)
from app.integrations.llm_cli.constants import (
    DEFAULT_EXEC_TIMEOUT_SEC as _DEFAULT_EXEC_TIMEOUT_SEC,
)
from app.integrations.llm_cli.constants import (
    MAX_EXEC_TIMEOUT_SEC as _MAX_EXEC_TIMEOUT_SEC,
)
from app.integrations.llm_cli.constants import (
    MIN_EXEC_TIMEOUT_SEC as _MIN_EXEC_TIMEOUT_SEC,
)
from app.integrations.llm_cli.env_overrides import (
    XAI_CLI_ENV_KEYS,
    nonempty_env_values,
)
from app.integrations.llm_cli.probe_utils import run_version_probe
from app.integrations.llm_cli.semver_utils import parse_semver_three_part
from app.integrations.llm_cli.timeout_utils import resolve_timeout_from_env

_PROBE_TIMEOUT_SEC = 5.0
_AUTH_HINT = "Run: grok login or set XAI_API_KEY."
# Conventional config directories where ``grok login`` persists its session token.
_GROK_CONFIG_DIRNAMES = (".grok-build", ".grok")
# Candidate session/credential filenames within a config dir (best-effort).
_GROK_SESSION_FILES = ("auth.json", "credentials.json", "session.json", "tokens.json")


def _resolve_exec_timeout_seconds() -> float:
    return resolve_timeout_from_env(
        env_key="GROK_CLI_TIMEOUT_SECONDS",
        default=_DEFAULT_EXEC_TIMEOUT_SEC,
        minimum=_MIN_EXEC_TIMEOUT_SEC,
        maximum=_MAX_EXEC_TIMEOUT_SEC,
    )


def _grok_env_overrides() -> dict[str, str]:
    """Subprocess env overrides: disable color and forward xAI API credentials."""
    env: dict[str, str] = {"NO_COLOR": "1"}
    env.update(nonempty_env_values(XAI_CLI_ENV_KEYS))
    return env


def _grok_session_authenticated() -> tuple[bool | None, str]:
    """Best-effort check for a persisted ``grok login`` session credential file."""
    for dirname in _GROK_CONFIG_DIRNAMES:
        config_dir = Path.home() / dirname
        for filename in _GROK_SESSION_FILES:
            creds = config_dir / filename
            try:
                if creds.exists() and creds.stat().st_size > 2:
                    return True, f"Authenticated via grok login session (~/{dirname}/{filename})."
            except OSError:
                return None, "Could not read Grok session credentials; auth state unclear."
    return False, f"Not logged in. {_AUTH_HINT}"


def _classify_grok_auth() -> tuple[bool | None, str]:
    """Return ``(logged_in, detail)`` for Grok Build CLI auth.

    Resolution order mirrors Grok's own credential precedence:
    1. ``XAI_API_KEY`` env → authenticated (works for headless/CI).
    2. Persisted ``grok login`` session file under a known config dir → authenticated.
    3. Otherwise not logged in (``False``); unreadable creds → unclear (``None``).
    """
    if os.environ.get("XAI_API_KEY", "").strip():
        return True, "Authenticated via XAI_API_KEY."
    return _grok_session_authenticated()


def _fallback_grok_paths() -> list[str]:
    return _default_cli_fallback_paths("grok")


class GrokCLIAdapter:
    """Non-interactive xAI Grok Build CLI (``grok -p``, headless mode, no TTY)."""

    name = "grok-cli"
    binary_env_key = "GROK_CLI_BIN"
    install_hint = "curl -fsSL https://x.ai/cli/install.sh | bash"
    auth_hint = _AUTH_HINT.removesuffix(".")
    min_version: str | None = None
    default_exec_timeout_sec = _DEFAULT_EXEC_TIMEOUT_SEC

    def _resolve_binary(self) -> str | None:
        return resolve_cli_binary(
            explicit_env_key="GROK_CLI_BIN",
            binary_names=_candidate_binary_names("grok"),
            fallback_paths=_fallback_grok_paths,
        )

    def _probe_binary(self, binary_path: str) -> CLIProbe:
        version_output, version_error = run_version_probe(
            binary_path,
            timeout_sec=_PROBE_TIMEOUT_SEC,
        )
        if version_error:
            return CLIProbe(
                installed=False,
                version=None,
                logged_in=None,
                bin_path=None,
                detail=version_error,
            )

        version = parse_semver_three_part(version_output or "")
        logged_in, auth_detail = _classify_grok_auth()
        return CLIProbe(
            installed=True,
            version=version,
            logged_in=logged_in,
            bin_path=binary_path,
            detail=auth_detail,
        )

    def detect(self) -> CLIProbe:
        binary = self._resolve_binary()
        if not binary:
            return CLIProbe(
                installed=False,
                version=None,
                logged_in=None,
                bin_path=None,
                detail=(
                    "Grok Build CLI not found on PATH or known install locations. "
                    f"Install with: {self.install_hint} or set GROK_CLI_BIN."
                ),
            )
        return self._probe_binary(binary)

    def build(
        self,
        *,
        prompt: str,
        model: str | None,
        workspace: str,
        reasoning_effort: str | None = None,
    ) -> CLIInvocation:
        # Grok Build headless mode does not expose a reasoning-effort flag; the
        # parameter is accepted for protocol compatibility and ignored.
        _ = reasoning_effort
        binary = self._resolve_binary()
        if not binary:
            raise RuntimeError(
                f"Grok Build CLI not found. {self.install_hint}"
                " or set GROK_CLI_BIN to the full binary path."
            )

        ws = (workspace or "").strip()
        cwd = str(Path(ws).expanduser()) if ws else os.getcwd()

        # `grok -p PROMPT` runs a single headless turn (no TTY). `--output-format
        # plain` yields the model's text answer for parse(); `--no-auto-update`
        # skips background update checks in scripted/CI environments. We
        # deliberately omit `--always-approve` so Grok never auto-executes its own
        # tools — OpenSRE drives tool use itself.
        argv: list[str] = [
            binary,
            "-p",
            prompt,
            "--output-format",
            "plain",
            "--no-auto-update",
        ]

        resolved_model = (model or "").strip()
        if resolved_model:
            argv.extend(["-m", resolved_model])

        # Forward xAI credentials explicitly rather than via the blanket prefix
        # allowlist, so XAI_API_KEY does not leak into other CLI adapters.
        env = _grok_env_overrides()

        return CLIInvocation(
            argv=tuple(argv),
            stdin=None,
            cwd=cwd,
            env=env,
            timeout_sec=_resolve_exec_timeout_seconds(),
        )

    def parse(self, *, stdout: str, stderr: str, returncode: int) -> str:
        del stderr, returncode
        return (stdout or "").strip()

    def explain_failure(self, *, stdout: str, stderr: str, returncode: int) -> str:
        err = (stderr or "").strip()
        out = (stdout or "").strip()
        bits = [f"grok -p exited with code {returncode}"]
        lowered = (err + "\n" + out).lower()
        if "unauthorized" in lowered or "401" in lowered or "not logged in" in lowered:
            bits.append(f"Authentication failed. {_AUTH_HINT}")
        elif err:
            bits.append(err[:2000])
        elif out:
            bits.append(out[:2000])
        return ". ".join(bits)
