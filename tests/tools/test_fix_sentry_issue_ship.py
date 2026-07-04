"""Tests for the Sentry issue-fix *shipping* path (branch + commit + push + PR).

Covers the git primitives against a real temp repo with a local bare remote, the
PR call with a mocked GitHub client, the ship orchestration, and the tool's
``open_pr`` wiring including every safety refusal (ship disabled, missing token,
protected branch, no changes, PR failure).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from integrations.github.client import GitHubApiError
from integrations.pi import PiCodingResult
from tools.cross_vendor.fix_sentry_issue import fix_sentry_issue, git_ops
from tools.cross_vendor.fix_sentry_issue.context import IssueContext
from tools.cross_vendor.fix_sentry_issue.errors import (
    ERR_COMMIT_FAILED,
    ERR_GITHUB_TOKEN,
    ERR_NO_CHANGES,
    ERR_PR_FAILED,
    ERR_PROTECTED_BRANCH,
    ERR_PUSH_FAILED,
    ERR_SHIP_DISABLED,
    FixIssueError,
)
from tools.cross_vendor.fix_sentry_issue.pr import PullRequest, open_pull_request
from tools.cross_vendor.fix_sentry_issue.ship import build_branch_name, ship_fix

_URL = "https://acme.sentry.io/issues/12345/"


# --------------------------------------------------------------------------- #
# git helpers
# --------------------------------------------------------------------------- #
def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(tmp_path: Path) -> Path:
    """A work tree on 'main' with an initial commit and a pushable bare 'origin'."""
    bare = tmp_path / "remote.git"
    work = tmp_path / "work"
    _git(tmp_path, "init", "--bare", str(bare))
    _git(tmp_path, "init", "-b", "main", str(work))
    _git(work, "config", "user.email", "t@example.com")
    _git(work, "config", "user.name", "Tester")
    (work / "README.md").write_text("hello\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-m", "init")
    _git(work, "remote", "add", "origin", str(bare))
    _git(work, "push", "-u", "origin", "main")
    _git(work, "remote", "set-head", "origin", "main")
    return work


def _success_result() -> PiCodingResult:
    return PiCodingResult(
        success=True,
        summary="Guard the None case in process_event.",
        changed_files=["app/handlers.py"],
        diff="diff --git a/app/handlers.py b/app/handlers.py\n",
        returncode=0,
    )


# --------------------------------------------------------------------------- #
# git_ops
# --------------------------------------------------------------------------- #
def test_git_ops_detects_repo_and_default_branch(tmp_path: Path) -> None:
    work = _init_repo(tmp_path)
    assert git_ops.is_git_repo(str(work)) is True
    assert git_ops.current_branch(str(work)) == "main"
    assert git_ops.default_branch(str(work)) == "main"
    assert git_ops.changed_paths(str(work)) == []


def test_git_ops_not_a_repo(tmp_path: Path) -> None:
    with pytest.raises(FixIssueError) as exc:
        git_ops.ensure_git_repo(str(tmp_path))
    assert exc.value.kind == "not_a_git_repo"


def test_changed_paths_reports_new_and_modified(tmp_path: Path) -> None:
    work = _init_repo(tmp_path)
    (work / "app").mkdir()
    (work / "app" / "handlers.py").write_text("x = 1\n")
    (work / "README.md").write_text("changed\n")
    paths = set(git_ops.changed_paths(str(work)))
    assert paths == {"app/handlers.py", "README.md"}


def test_changed_paths_handles_quoted_filenames(tmp_path: Path) -> None:
    work = _init_repo(tmp_path)
    # A space + non-ASCII byte makes git C-quote the path in default porcelain output.
    weird = "wéird name.txt"
    (work / weird).write_text("x\n")

    # -z returns the path verbatim (unquoted), so downstream git ops match it.
    assert weird in git_ops.changed_paths(str(work))
    assert git_ops.file_fingerprints(str(work), [weird])[weird]  # hashable

    git_ops.create_branch(str(work), "opensre/sentry-fix-1-x", base_default="main")
    git_ops.commit_paths(str(work), [weird], "add weird")
    # The odd path was actually committed (it is no longer reported as changed).
    assert weird not in git_ops.changed_paths(str(work))


@pytest.mark.parametrize("branch", ["main", "master", "develop", "trunk", "", "   "])
def test_assert_not_protected_rejects_base_and_empty(branch: str) -> None:
    with pytest.raises(FixIssueError) as exc:
        git_ops.assert_not_protected(branch, protected_extra="main")
    assert exc.value.kind == ERR_PROTECTED_BRANCH


def test_assert_not_protected_rejects_resolved_default() -> None:
    with pytest.raises(FixIssueError):
        git_ops.assert_not_protected("release-1.0", protected_extra="release-1.0")


def test_create_branch_refuses_protected(tmp_path: Path) -> None:
    work = _init_repo(tmp_path)
    with pytest.raises(FixIssueError) as exc:
        git_ops.create_branch(str(work), "main", base_default="main")
    assert exc.value.kind == ERR_PROTECTED_BRANCH
    assert git_ops.current_branch(str(work)) == "main"  # still on base


def test_branch_commit_push_roundtrip(tmp_path: Path) -> None:
    work = _init_repo(tmp_path)
    (work / "app").mkdir()
    (work / "app" / "handlers.py").write_text("x = 1\n")
    branch = "opensre/sentry-fix-12345-abc"

    git_ops.create_branch(str(work), branch, base_default="main")
    git_ops.commit_paths(str(work), ["app/handlers.py"], "fix: something")
    git_ops.push_branch(str(work), branch, base_default="main")

    assert git_ops.current_branch(str(work)) == branch
    # The branch exists on the remote; base branch is untouched.
    remote_branches = subprocess.run(
        ["git", "branch", "-r"], cwd=work, capture_output=True, text=True
    ).stdout
    assert f"origin/{branch}" in remote_branches


def test_commit_paths_isolates_to_given_files(tmp_path: Path) -> None:
    work = _init_repo(tmp_path)
    # An unrelated tracked file, committed to the base.
    (work / "other.txt").write_text("orig\n")
    _git(work, "add", "other.txt")
    _git(work, "commit", "-m", "add other")

    # Pi's fix (a new file) alongside unrelated WIP: README modified (unstaged)
    # and other.txt modified AND staged.
    (work / "app").mkdir()
    (work / "app" / "handlers.py").write_text("fix\n")
    (work / "README.md").write_text("wip unstaged\n")
    (work / "other.txt").write_text("wip staged\n")
    _git(work, "add", "other.txt")

    git_ops.commit_paths(str(work), ["app/handlers.py"], "fix: only pi file")

    # The commit contains ONLY Pi's file, not the unrelated WIP.
    committed = subprocess.run(
        ["git", "show", "--name-only", "--pretty=format:", "HEAD"],
        cwd=work,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert committed == ["app/handlers.py"]
    # The unrelated WIP is left uncommitted in the tree.
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=work, capture_output=True, text=True
    ).stdout
    assert "README.md" in status
    assert "other.txt" in status


def test_token_auth_env_injects_header_without_leaking_token() -> None:
    env = git_ops._token_auth_env("secret-token")
    assert env["GIT_CONFIG_KEY_0"] == "http.extraheader"
    header = env["GIT_CONFIG_VALUE_0"]
    assert header.startswith("Authorization: Basic ")
    # The raw token must never appear verbatim (it's base64'd behind the header).
    assert "secret-token" not in header


def test_push_branch_uses_token_env_when_provided(tmp_path: Path) -> None:
    work = _init_repo(tmp_path)
    captured: dict[str, Any] = {}

    def _fake_run_git(_ws: str, *args: str, env: Any = None) -> Any:
        captured["args"] = args
        captured["env"] = env
        return subprocess.CompletedProcess(args=list(args), returncode=0, stdout="", stderr="")

    with patch.object(git_ops, "_run_git", _fake_run_git):
        git_ops.push_branch(
            str(work), "opensre/sentry-fix-1-x", base_default="main", github_token="tok"
        )

    assert captured["args"][0] == "push"
    assert captured["env"] is not None
    assert captured["env"]["GIT_CONFIG_KEY_0"] == "http.extraheader"


def test_push_branch_no_env_without_token(tmp_path: Path) -> None:
    work = _init_repo(tmp_path)
    captured: dict[str, Any] = {}

    def _fake_run_git(_ws: str, *args: str, env: Any = None) -> Any:
        captured["env"] = env
        return subprocess.CompletedProcess(args=list(args), returncode=0, stdout="", stderr="")

    with patch.object(git_ops, "_run_git", _fake_run_git):
        git_ops.push_branch(str(work), "opensre/sentry-fix-1-x", base_default="main")

    assert captured["env"] is None


# --------------------------------------------------------------------------- #
# pr.open_pull_request
# --------------------------------------------------------------------------- #
_SCOPE = "tools.cross_vendor.fix_sentry_issue.pr.detect_git_remote_repo_scope"
_CLIENT = "tools.cross_vendor.fix_sentry_issue.pr.GitHubRestClient"


@patch(_SCOPE, return_value=("acme", "app"))
def test_open_pull_request_builds_payload(mock_scope: MagicMock) -> None:
    client = MagicMock()
    client.request.return_value = {"html_url": "https://github.com/acme/app/pull/7", "number": 7}
    with patch(_CLIENT, return_value=client):
        pr = open_pull_request(
            "/ws",
            head_branch="opensre/sentry-fix-1-x",
            base_branch="main",
            title="fix: resolve Sentry issue 1",
            body="body",
            github_token="tok",
        )
    assert pr == PullRequest(url="https://github.com/acme/app/pull/7", number=7)
    method, path = client.request.call_args.args
    body = client.request.call_args.kwargs["body"]
    assert (method, path) == ("POST", "repos/acme/app/pulls")
    assert body["head"] == "opensre/sentry-fix-1-x"
    assert body["base"] == "main"


def test_open_pull_request_requires_token() -> None:
    with patch.dict("os.environ", {}, clear=True), pytest.raises(FixIssueError) as exc:
        open_pull_request(
            "/ws", head_branch="b", base_branch="main", title="t", body="b", github_token=""
        )
    assert exc.value.kind == ERR_GITHUB_TOKEN


@patch(_SCOPE, return_value=None)
def test_open_pull_request_unknown_repo_scope(_mock_scope: MagicMock) -> None:
    with pytest.raises(FixIssueError) as exc:
        open_pull_request(
            "/ws", head_branch="b", base_branch="main", title="t", body="b", github_token="tok"
        )
    assert exc.value.kind == ERR_PR_FAILED


@patch(_SCOPE, return_value=("acme", "app"))
def test_open_pull_request_maps_api_error(_mock_scope: MagicMock) -> None:
    client = MagicMock()
    client.request.side_effect = GitHubApiError("validation failed", status_code=422)
    with patch(_CLIENT, return_value=client), pytest.raises(FixIssueError) as exc:
        open_pull_request(
            "/ws", head_branch="b", base_branch="main", title="t", body="b", github_token="tok"
        )
    assert exc.value.kind == ERR_PR_FAILED


# --------------------------------------------------------------------------- #
# ship.ship_fix
# --------------------------------------------------------------------------- #
def test_build_branch_name_is_namespaced(tmp_path: Path) -> None:
    work = _init_repo(tmp_path)
    name = build_branch_name(str(work), "12345")
    assert name.startswith("opensre/sentry-fix-12345-")
    assert name != "main"


def test_build_branch_name_slugs_weird_ids(tmp_path: Path) -> None:
    work = _init_repo(tmp_path)
    assert build_branch_name(str(work), "a/b c!").startswith("opensre/sentry-fix-a-b-c-")


def test_ship_fix_no_changes_raises(tmp_path: Path) -> None:
    work = _init_repo(tmp_path)  # clean tree
    with pytest.raises(FixIssueError) as exc:
        ship_fix(str(work), issue_id="12345", sentry_url=_URL, result=_success_result())
    assert exc.value.kind == ERR_NO_CHANGES


def test_ship_fix_full_roundtrip(tmp_path: Path) -> None:
    work = _init_repo(tmp_path)
    (work / "app").mkdir()
    (work / "app" / "handlers.py").write_text("x = 1\n")

    with patch(
        "tools.cross_vendor.fix_sentry_issue.ship.open_pull_request",
        return_value=PullRequest(url="https://github.com/acme/app/pull/9", number=9),
    ) as mock_pr:
        ship = ship_fix(str(work), issue_id="12345", sentry_url=_URL, result=_success_result())

    assert ship.branch_name.startswith("opensre/sentry-fix-12345-")
    assert ship.pr.number == 9
    # PR opened from the new branch into the base branch.
    assert mock_pr.call_args.kwargs["base_branch"] == "main"
    assert mock_pr.call_args.kwargs["head_branch"] == ship.branch_name
    # The fix was committed onto the feature branch, not main.
    assert git_ops.current_branch(str(work)) == ship.branch_name
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"], cwd=work, capture_output=True, text=True
    ).stdout
    assert "12345" in log


def test_ship_fix_tags_branch_on_push_failure(tmp_path: Path) -> None:
    work = _init_repo(tmp_path)
    (work / "app").mkdir()
    (work / "app" / "handlers.py").write_text("x = 1\n")

    # Push fails after the branch is created and the fix is committed.
    with (
        patch(
            "tools.cross_vendor.fix_sentry_issue.ship.push_branch",
            side_effect=FixIssueError(ERR_PUSH_FAILED, "remote rejected"),
        ),
        pytest.raises(FixIssueError) as exc,
    ):
        ship_fix(str(work), issue_id="12345", sentry_url=_URL, result=_success_result())

    # The error carries the branch that holds the committed fix, for manual recovery.
    assert exc.value.branch_name is not None
    assert exc.value.branch_name.startswith("opensre/sentry-fix-12345-")
    # The commit really exists on that branch.
    assert git_ops.current_branch(str(work)) == exc.value.branch_name
    committed = subprocess.run(
        ["git", "show", "--name-only", "--pretty=format:", "HEAD"],
        cwd=work,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert committed == ["app/handlers.py"]


def test_ship_fix_tags_branch_on_commit_failure(tmp_path: Path) -> None:
    work = _init_repo(tmp_path)
    (work / "app").mkdir()
    (work / "app" / "handlers.py").write_text("x = 1\n")

    # Commit fails after the branch is created (e.g. hook rejection / missing identity).
    with (
        patch(
            "tools.cross_vendor.fix_sentry_issue.ship.commit_paths",
            side_effect=FixIssueError(ERR_COMMIT_FAILED, "git commit failed"),
        ),
        pytest.raises(FixIssueError) as exc,
    ):
        ship_fix(str(work), issue_id="12345", sentry_url=_URL, result=_success_result())

    # Even though nothing was committed, the workspace switched to the new branch,
    # so the error must surface it for recovery instead of reporting None.
    assert exc.value.branch_name is not None
    assert exc.value.branch_name.startswith("opensre/sentry-fix-12345-")
    assert git_ops.current_branch(str(work)) == exc.value.branch_name


def _baseline(work: Path) -> dict[str, str]:
    return git_ops.file_fingerprints(str(work), git_ops.changed_paths(str(work)))


def test_ship_fix_commits_only_pi_files_not_unrelated_wip(tmp_path: Path) -> None:
    work = _init_repo(tmp_path)
    # Unrelated developer WIP present BEFORE the Pi run -> captured in the baseline.
    (work / "wip.txt").write_text("do not ship me\n")
    baseline = _baseline(work)
    assert list(baseline) == ["wip.txt"]

    # Pi's fix, created during the run (wip.txt is left untouched).
    (work / "app").mkdir()
    (work / "app" / "handlers.py").write_text("x = 1\n")

    result = PiCodingResult(
        success=True, summary="s", changed_files=["app/handlers.py"], diff="", returncode=0
    )
    with patch(
        "tools.cross_vendor.fix_sentry_issue.ship.open_pull_request",
        return_value=PullRequest(url="https://github.com/acme/app/pull/9", number=9),
    ):
        ship = ship_fix(
            str(work), issue_id="12345", sentry_url=_URL, result=result, baseline=baseline
        )

    committed = subprocess.run(
        ["git", "show", "--name-only", "--pretty=format:", ship.branch_name],
        cwd=work,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert committed == ["app/handlers.py"]  # untouched wip.txt was NOT swept in
    assert (
        "wip.txt"
        in subprocess.run(
            ["git", "status", "--porcelain"], cwd=work, capture_output=True, text=True
        ).stdout
    )


def test_ship_fix_includes_pi_edit_to_already_dirty_file(tmp_path: Path) -> None:
    work = _init_repo(tmp_path)
    # README.md is tracked; a developer has uncommitted WIP in it before the run.
    (work / "README.md").write_text("dev wip\n")
    baseline = _baseline(work)
    assert list(baseline) == ["README.md"]

    # Pi ALSO edits the same file — its change must NOT be dropped just because the
    # file was already dirty.
    (work / "README.md").write_text("dev wip + pi fix\n")

    result = PiCodingResult(
        success=True, summary="s", changed_files=["README.md"], diff="", returncode=0
    )
    with patch(
        "tools.cross_vendor.fix_sentry_issue.ship.open_pull_request",
        return_value=PullRequest(url="https://github.com/acme/app/pull/9", number=9),
    ):
        ship = ship_fix(
            str(work), issue_id="12345", sentry_url=_URL, result=result, baseline=baseline
        )

    committed = subprocess.run(
        ["git", "show", "--name-only", "--pretty=format:", ship.branch_name],
        cwd=work,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert committed == ["README.md"]  # Pi's edit to the dirty file is shipped


def test_ship_fix_no_new_changes_over_baseline_raises(tmp_path: Path) -> None:
    work = _init_repo(tmp_path)
    # Everything dirty was already there before Pi and left untouched (all baseline).
    (work / "wip.txt").write_text("pre-existing\n")
    baseline = _baseline(work)

    with pytest.raises(FixIssueError) as exc:
        ship_fix(
            str(work),
            issue_id="12345",
            sentry_url=_URL,
            result=_success_result(),
            baseline=baseline,
        )
    assert exc.value.kind == ERR_NO_CHANGES


# --------------------------------------------------------------------------- #
# tool run() with open_pr
# --------------------------------------------------------------------------- #
_CTX = IssueContext(issue_id="12345", task="Sentry issue task")
_TOOL_GATHER = "tools.cross_vendor.fix_sentry_issue.gather_issue_context"
_TOOL_CLI = "tools.cross_vendor.fix_sentry_issue.ensure_cli_ready"
_TOOL_RUNFIX = "tools.cross_vendor.fix_sentry_issue.run_fix"
_TOOL_SHIP = "tools.cross_vendor.fix_sentry_issue.run_ship"
_TOOL_PRE = "tools.cross_vendor.fix_sentry_issue.pre_pi_changes"


@patch(_TOOL_RUNFIX)
@patch(_TOOL_CLI)
@patch(_TOOL_GATHER, return_value=_CTX)
def test_run_open_pr_disabled_fails_fast(
    _gather: MagicMock, _cli: MagicMock, mock_runfix: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PI_ISSUE_FIX_ENABLED", "1")
    monkeypatch.delenv("PI_ISSUE_FIX_SHIP_ENABLED", raising=False)
    out = fix_sentry_issue.run(sentry_url=_URL, open_pr=True)
    assert out["error_kind"] == ERR_SHIP_DISABLED
    mock_runfix.assert_not_called()  # never spend a Pi run if we can't ship
    # The issue is already resolved, so its id must survive the ship-gate failure.
    assert out["issue_id"] == _CTX.issue_id
    # Early-gate failures must still carry the full, stable output shape so
    # callers can read these keys without KeyError.
    for key in ("branch_name", "pr_url", "pr_number", "summary", "changed_files", "diff"):
        assert key in out
    assert out["pr_url"] is None and out["branch_name"] is None and out["pr_number"] is None


@patch(_TOOL_RUNFIX)
@patch(_TOOL_CLI)
@patch(_TOOL_GATHER, return_value=_CTX)
def test_run_open_pr_missing_token_fails_fast(
    _gather: MagicMock, _cli: MagicMock, mock_runfix: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PI_ISSUE_FIX_ENABLED", "1")
    monkeypatch.setenv("PI_ISSUE_FIX_SHIP_ENABLED", "1")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    out = fix_sentry_issue.run(sentry_url=_URL, open_pr=True)
    assert out["error_kind"] == ERR_GITHUB_TOKEN
    assert out["issue_id"] == _CTX.issue_id  # resolved id preserved on ship-gate failure
    mock_runfix.assert_not_called()


@patch(_TOOL_PRE, return_value={"pre_existing_wip.txt": "deadbeef"})
@patch(_TOOL_SHIP)
@patch(_TOOL_RUNFIX, return_value=_success_result())
@patch("tools.cross_vendor.fix_sentry_issue.ensure_ship_ready")
@patch(_TOOL_CLI)
@patch(_TOOL_GATHER, return_value=_CTX)
def test_run_open_pr_success_returns_pr(
    _gather: MagicMock,
    _cli: MagicMock,
    _ship_ready: MagicMock,
    _runfix: MagicMock,
    mock_ship: MagicMock,
    _pre: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.cross_vendor.fix_sentry_issue.ship import ShipResult

    mock_ship.return_value = ShipResult(
        branch_name="opensre/sentry-fix-12345-abc",
        pr=PullRequest(url="https://github.com/acme/app/pull/9", number=9),
    )
    monkeypatch.setenv("PI_ISSUE_FIX_ENABLED", "1")
    monkeypatch.setenv("PI_ISSUE_FIX_SHIP_ENABLED", "1")
    out = fix_sentry_issue.run(sentry_url=_URL, open_pr=True)
    assert out["success"] is True
    assert out["error_kind"] is None
    assert out["pr_url"] == "https://github.com/acme/app/pull/9"
    assert out["pr_number"] == 9
    assert out["branch_name"] == "opensre/sentry-fix-12345-abc"
    # The pre-Pi fingerprint snapshot is threaded to the ship step as the baseline.
    assert mock_ship.call_args.kwargs["baseline"] == {"pre_existing_wip.txt": "deadbeef"}


@patch(_TOOL_PRE, return_value=())
@patch(
    _TOOL_SHIP,
    side_effect=FixIssueError(
        "push_failed", "remote rejected", branch_name="opensre/sentry-fix-12345-xyz"
    ),
)
@patch(_TOOL_RUNFIX, return_value=_success_result())
@patch("tools.cross_vendor.fix_sentry_issue.ensure_ship_ready")
@patch(_TOOL_CLI)
@patch(_TOOL_GATHER, return_value=_CTX)
def test_run_open_pr_ship_failure_keeps_diff_and_branch(
    _gather: MagicMock,
    _cli: MagicMock,
    _ship_ready: MagicMock,
    _runfix: MagicMock,
    _ship: MagicMock,
    _pre: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PI_ISSUE_FIX_ENABLED", "1")
    monkeypatch.setenv("PI_ISSUE_FIX_SHIP_ENABLED", "1")
    out = fix_sentry_issue.run(sentry_url=_URL, open_pr=True)
    assert out["success"] is False
    assert out["error_kind"] == "push_failed"
    assert out["pr_url"] is None
    assert "diff --git" in out["diff"]  # the fix is preserved for manual shipping
    assert out["changed_files"] == ["app/handlers.py"]
    # A post-commit failure surfaces the branch so the user can push it manually.
    assert out["branch_name"] == "opensre/sentry-fix-12345-xyz"


@patch(_TOOL_SHIP)
@patch(_TOOL_RUNFIX, return_value=_success_result())
@patch(_TOOL_CLI)
@patch(_TOOL_GATHER, return_value=_CTX)
def test_run_without_open_pr_never_ships(
    _gather: MagicMock,
    _cli: MagicMock,
    _runfix: MagicMock,
    mock_ship: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PI_ISSUE_FIX_ENABLED", "1")
    out = fix_sentry_issue.run(sentry_url=_URL)
    assert out["success"] is True
    assert out["pr_url"] is None
    assert out["branch_name"] is None
    mock_ship.assert_not_called()
