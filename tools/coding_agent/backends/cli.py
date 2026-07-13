"""CLI-subprocess coding backend: adapts a ``CodingCLIAdapter`` to ``CodingBackend``.

This is the **only** module in the coding path that knows about argv, subprocesses,
and exit codes. It builds the agent's own prompt via the adapter, runs the resulting
invocation under the polled executor, and translates CLI-shaped failures (non-zero
exit, provider-limit signatures, spawn errors) into a transport-neutral
:class:`~tools.coding_agent.backends.base.BackendOutcome`.

Every CLI agent — Pi today, codex/claude-code once their adapters grow a coding
mode — reaches the runner through this one backend. A non-CLI agent (OpenClaw over
MCP) gets its own sibling backend and never touches this file.
"""

from __future__ import annotations

from dataclasses import replace

from integrations.llm_cli.base import CodingCLIAdapter
from integrations.llm_cli.polled_runner import ProcessOutcome, run_polled_process
from platform.masking import MaskingContext, MaskingPolicy
from tools.coding_agent.backends.base import BackendOutcome, BackendProbe

_MAX_OUTPUT_CHARS = 8000

# Provider-side limit/error signatures agents print (often to stdout, exit 0). These
# are specific error phrases — NOT bare words like "quota" — so a task that edits
# quota/rate-limit code is not misread as a provider failure.
_LIMIT_MARKERS: tuple[str, ...] = (
    "resource_exhausted",
    "too many requests",
    "exceeded your current quota",
    "quota exceeded",
    "rate limit exceeded",
    "rate_limit_exceeded",
    "credit balance is too low",
    '"code":429',
    '"code": 429',
    '"code":413',
    '"code": 413',
)


class CLICodingBackend:
    """Runs a coding task by driving a CLI agent as a polled subprocess."""

    def __init__(self, adapter: CodingCLIAdapter, *, name: str) -> None:
        self._adapter = adapter
        self.name = name

    def detect(self) -> BackendProbe:
        """Installed and authenticated? Auth ``None`` (unclear) counts as ready —
        the run itself will surface a clear error if it turns out not to be."""
        probe = self._adapter.detect()
        if not probe.installed or probe.logged_in is False:
            return BackendProbe(ready=False, detail=probe.detail)
        return BackendProbe(ready=True, detail=probe.detail)

    def run(
        self,
        task: str,
        *,
        workspace: str,
        model: str | None,
        timeout_sec: float,
    ) -> BackendOutcome:
        # The agent owns its prompt (safety rules included), so the tool layer
        # never carries one.
        prompt = self._adapter.build_coding_prompt(task)
        try:
            invocation = self._adapter.build(
                prompt=prompt, model=model, workspace=workspace, coding_mode=True
            )
        except RuntimeError as exc:
            # e.g. binary not found at build time — an expected pre-flight failure.
            return BackendOutcome(summary="", error=str(exc), exit_code=-1)

        # Use the configured coding timeout, not the adapter's default (which is
        # tuned for short answer-mode calls).
        outcome = run_polled_process(replace(invocation, timeout_sec=timeout_sec))
        if outcome.spawn_error:
            return BackendOutcome(summary="", error=outcome.spawn_error, exit_code=-1)
        return self._classify(outcome, timeout_sec=timeout_sec)

    def _classify(self, outcome: ProcessOutcome, *, timeout_sec: float) -> BackendOutcome:
        """Turn a finished subprocess into a neutral outcome.

        Free-text is masked (an agent may echo env/secrets). This decides only whether
        the *run* failed — whether it *achieved* anything is the runner's call, since
        that depends on the git diff.
        """
        masker = MaskingContext(MaskingPolicy.from_env())
        out_text = outcome.stdout.strip()
        err_text = outcome.stderr.strip()
        summary = masker.mask(out_text[:_MAX_OUTPUT_CHARS])
        detail = masker.mask(
            (err_text or out_text or f"{self.name} exited with code {outcome.returncode}")[
                :_MAX_OUTPUT_CHARS
            ]
        )

        if outcome.timed_out:
            return BackendOutcome(
                summary=summary,
                error=f"{self.name} timed out after {timeout_sec:.0f}s",
                timed_out=True,
                exit_code=outcome.returncode,
            )

        # A non-zero exit is a definite failure, whatever the tree looks like.
        if outcome.returncode != 0:
            return BackendOutcome(summary=summary, error=detail, exit_code=outcome.returncode)

        # Agents print provider errors (e.g. a 429) to *stdout* and still exit 0, so
        # scan for limit signatures anyway — but report them as a *signal*, not a
        # failure. An agent that successfully edits rate-limiting code echoes the same
        # phrases; only the runner, which knows whether the tree changed, can tell the
        # two apart.
        lowered = f"{out_text}\n{err_text}".lower()
        if any(marker in lowered for marker in _LIMIT_MARKERS):
            return BackendOutcome(
                summary=summary,
                limit_signal=True,
                limit_detail=detail,
                exit_code=outcome.returncode,
            )

        return BackendOutcome(summary=summary, exit_code=outcome.returncode)
