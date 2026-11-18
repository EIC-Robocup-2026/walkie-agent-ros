"""bwrap wraps each worker, and a broken bwrap stops the run instead of dropping the wall."""

import os
import stat
import subprocess
from pathlib import Path

import pytest

from walkie_agent.capabilities.fake import FakeRobot
from walkie_agent.scripting.driver import ScriptRunner, allow_list
from walkie_agent.scripting.sandbox import SandboxUnavailable, bwrap_command, open_pool
from walkie_agent.tools.sets import GPSR


def sandboxed(program, *args):
    """Run `program` under the same bwrap arguments as a worker, with /proc added for the probe."""
    command = bwrap_command(program, worker="/probe")
    return subprocess.run(
        [*command[:-1], "--proc", "/proc", command[-1], *args],
        capture_output=True,
        text=True,
        env={**os.environ, "SECRET_PROBE": "leaked"},
        timeout=20,
    )


def test_the_isolation_flags_are_the_plans():
    assert bwrap_command("/bin/x")[:7] == [
        "bwrap",
        "--clearenv",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
    ]


def test_the_environment_is_empty_in_the_sandbox():
    done = sandboxed("/usr/bin/env")

    assert done.returncode == 0, done.stderr
    assert "SECRET_PROBE" not in done.stdout
    assert "OPENAI_API_KEY" not in done.stdout
    assert "HOME" not in done.stdout


@pytest.mark.parametrize("path", ["/home", "/etc", "/root", "/var", "/tmp", str(Path.home())])
def test_host_directories_do_not_exist_in_the_sandbox(path):
    done = sandboxed("/bin/sh", "-c", f"test -e {path}")

    assert done.returncode == 1, done.stderr


def test_the_sandbox_has_its_own_network_namespace():
    done = sandboxed("/usr/bin/readlink", "/proc/self/ns/net")

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() != os.readlink("/proc/self/ns/net")


def _descendants(pid):
    children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    return [pid, *(found for child in children for found in _descendants(int(child)))]


def test_a_real_worker_runs_a_script_in_the_sandbox():
    robot = FakeRobot(seed=1)
    with open_pool(max_processes=1) as pool:
        with pool.checkout() as session:
            worker = session.worker_pid or 0
            session.feed_run("1 + 1")
            tree = _descendants(worker)
            # The deepest process is Monty itself. bwrap's outer monitor keeps the
            # host's environment, but nothing in the sandbox's namespaces can see it.
            # bwrap itself sets PWD, and that is all the worker gets.
            assert Path(f"/proc/{tree[-1]}/environ").read_bytes() == b"PWD=/\0"
            assert os.readlink(f"/proc/{tree[-1]}/ns/net") != os.readlink("/proc/self/ns/net")
            assert os.readlink(f"/proc/{tree[-1]}/ns/mnt") != os.readlink("/proc/self/ns/mnt")
        result = ScriptRunner(pool).run('go_to("shelf")', allow_list(GPSR, robot.ctx))

    assert result.ok, result.error
    assert robot.snapshot()["at"] == "shelf"


def test_a_missing_bwrap_fails_closed():
    with pytest.raises(SandboxUnavailable, match="not installed"):
        with open_pool(bwrap="no-such-bwrap", max_processes=1):
            pytest.fail("The pool opened without a sandbox.")


def test_a_broken_bwrap_fails_closed(tmp_path):
    broken = tmp_path / "bwrap"
    broken.write_text("#!/bin/sh\nexit 1\n")
    broken.chmod(stat.S_IRWXU)

    with pytest.raises(SandboxUnavailable, match="could not start"):
        with open_pool(bwrap=str(broken), max_processes=1):
            pytest.fail("The pool opened without a sandbox.")


def test_the_wrapper_is_gone_after_the_pool_closes(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))

    with open_pool(max_processes=1):
        assert [path.name for path in tmp_path.iterdir() if path.name.startswith("mahoraga-bwrap-")]

    assert not [path.name for path in tmp_path.iterdir() if path.name.startswith("mahoraga-bwrap-")]


def test_the_pool_runs_without_bwrap_only_when_asked():
    with open_pool(sandbox=False, max_processes=1) as pool:
        with pool.checkout() as session:
            assert session.feed_run("1 + 1") == 2
