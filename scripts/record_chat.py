"""Record a real chat.py session as a terminal transcript.

`chat.py` is an interactive terminal loop. Piping prompts into it works, but the
shell flushes every line at once, so the captured output is out of order and the
prompts are not interleaved with the replies. This driver runs `chat.py` in a
pseudo-terminal and types one line at a time, waiting for each "You: " prompt, so
the captured text is exactly what a person sitting at the terminal would see.

The replies are real model output: this only supplies the keystrokes. `chat.py`
itself is unmodified and writes its own JSON transcript as usual.

Run:  python scripts/record_chat.py --model llm_runs/<run>/model.pt \
          --transcript results/chat_transcript.json \
          --recording results/chat_recording.txt
"""

from __future__ import annotations

import argparse
import os
import pty
import re
import select
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROMPTS = [
    "the surgeon discussed the",
    "the opposite of tall is",
    "a duckling grows into a",
    "one horse",
    "what is the capital of france ?",
    "the team discussed the customer and the service at the store and then the report "
    "about the merchandise explains the quality in detail and our office has a question "
    "about the new application and the security update and the tutor reviewed the lesson "
    "and the course at the school and the surgeon explained the treatment",
]


def run(command: list[str], lines: list[str], idle_timeout: float = 30.0) -> str:
    """Run `command` in a pty, sending one line each time it asks for input."""
    pending = list(lines) + ["/quit"]
    captured: list[str] = []
    pid, master = pty.fork()
    if pid == 0:  # child
        os.execvp(command[0], command)
    buffer = ""
    try:
        while True:
            ready, _, _ = select.select([master], [], [], idle_timeout)
            if not ready:
                break
            try:
                chunk = os.read(master, 4096)
            except OSError:
                break
            if not chunk:
                break
            text = chunk.decode("utf-8", "replace")
            captured.append(text)
            buffer += text
            # chat.py's input() prompt is exactly "You: "
            if buffer.endswith("You: ") and pending:
                line = pending.pop(0)
                os.write(master, (line + "\n").encode())
                buffer = ""
    finally:
        os.close(master)
        os.waitpid(pid, 0)
    return "".join(captured)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    if args.transcript.exists():
        parser.error(f"{args.transcript} exists; chat.py needs a fresh transcript path.")

    output = run([args.python, str(ROOT / "chat.py"),
                  "--model", str(args.model), "--transcript", str(args.transcript)],
                 PROMPTS)
    # A pty writes CRLF; normalise so the file reads cleanly in a browser.
    cleaned = re.sub(r"\r\n?", "\n", output)
    args.recording.parent.mkdir(parents=True, exist_ok=True)
    args.recording.write_text(cleaned, encoding="utf-8")
    print(cleaned)
    print(f"[recorded to {args.recording}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
