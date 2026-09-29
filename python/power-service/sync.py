#!/usr/bin/env python3
"""
sync.py

Sends the power log to the online dashboard once the Pi can reach it.

pmic-web.py writes to disk every second whether or not anything is listening,
so a stretch with no internet costs no data. What was missing was a way to get
that backlog out afterwards. This walks the files, remembers how far it got,
and only moves that mark once the other end has said it stored the batch. Pull
the plug halfway through and the worst that happens is the same rows arrive
twice, which is why every row carries its own timestamp: the dashboard is meant
to treat (device, kind, t) as the identity of a reading and ignore repeats.

Nothing here decides when the Pi is online. It simply tries, and treats a
failure as "not now" rather than an error, because on this machine being
unreachable is the normal state rather than the exception.

Configure with environment variables, so no address or token lives in the code:

    SYNC_URL      where to post, e.g. https://dashboard.example/api/ingest
    SYNC_TOKEN    sent as "Authorization: Bearer ..." when set
    SYNC_DEVICE   which bot this is, default "solarpi"
    SYNC_EVERY    seconds between attempts, default 300
    POWER_DATA_DIR  where the logs are, default ~/solarbot/data/power

Run it once to push whatever is waiting and stop:

    python3 sync.py --once

Or see what it would send without sending anything:

    python3 sync.py --dry-run
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

HOME = os.path.expanduser("~")
DATA_DIR = os.environ.get("POWER_DATA_DIR",
                          os.path.join(HOME, "solarbot", "data", "power"))
STATE_FILE = os.path.join(DATA_DIR, ".sync-state.json")

URL = os.environ.get("SYNC_URL", "")
TOKEN = os.environ.get("SYNC_TOKEN", "")
DEVICE = os.environ.get("SYNC_DEVICE", "solarpi")
EVERY = int(os.environ.get("SYNC_EVERY", "300"))

# One batch is capped so a month of backlog goes up in pieces rather than as a
# single enormous request that a flaky link will never finish.
MAX_ROWS = 2000

# The header of power-DATE.csv, in file order.
POWER_FIELDS = ["iso", "t", "total_w", "cpu_w", "battery_pct", "battery_v",
                "plugged", "panel_v", "panel_a", "panel_w"]


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def load_state():
    """How far we got in each file, as {filename: bytes consumed}."""
    try:
        with open(STATE_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_state(state):
    """Write via a temporary file so a power cut cannot leave half a state."""
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=1, sort_keys=True)
    os.replace(tmp, STATE_FILE)


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def read_new(path, offset):
    """Return (lines, new_offset) for whatever was appended since last time.

    Reads bytes rather than lines so the mark survives a restart exactly, and
    stops at the last newline so a row still being written is left for the
    next pass instead of being sent in half.
    """
    size = os.path.getsize(path)
    if offset >= size:
        return [], offset
    with open(path, "rb") as fh:
        fh.seek(offset)
        blob = fh.read(size - offset)
    cut = blob.rfind(b"\n")
    if cut == -1:
        return [], offset
    text = blob[:cut].decode("utf-8", "replace")
    return [ln for ln in text.split("\n") if ln.strip()], offset + cut + 1


def power_rows(lines):
    """Turn CSV lines into dicts, skipping the header if it comes past."""
    out = []
    for ln in lines:
        parts = ln.split(",")
        if not parts or parts[0] == "time":
            continue
        row = dict(zip(POWER_FIELDS, parts))
        if not row.get("t"):
            continue
        row["t"] = num(row["t"])
        for k in ("total_w", "cpu_w", "battery_pct", "battery_v",
                  "panel_v", "panel_a", "panel_w"):
            row[k] = num(row.get(k))
        row["plugged"] = row.get("plugged") or None
        out.append(row)
    return out


def question_rows(lines):
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def post(kind, rows, dry_run=False):
    """Send one batch. True means it is safely stored and the mark may move."""
    body = json.dumps({"device": DEVICE, "kind": kind, "rows": rows}).encode()
    if dry_run:
        log("  zou %d %s-regels sturen (%d kB)" % (len(rows), kind, len(body) / 1024))
        return True
    req = urllib.request.Request(URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if TOKEN:
        req.add_header("Authorization", "Bearer " + TOKEN)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            ok = 200 <= r.status < 300
            if not ok:
                log("  dashboard antwoordde %s" % r.status)
            return ok
    except urllib.error.HTTPError as e:
        # A refusal is the server's decision and retrying will not change it,
        # but the mark stays put so nothing is silently dropped.
        log("  geweigerd: HTTP %s" % e.code)
        return False
    except Exception as e:
        # No route, no DNS, timeout: the ordinary offline case.
        log("  niet bereikbaar (%s)" % type(e).__name__)
        return False


def sources():
    """Every log file, oldest first, so the backlog arrives in order."""
    out = []
    try:
        names = sorted(os.listdir(DATA_DIR))
    except OSError:
        return out
    for n in names:
        if n.startswith("power-") and n.endswith(".csv"):
            out.append((n, "power"))
    if "questions.jsonl" in names:
        out.append(("questions.jsonl", "questions"))
    return out


def run_once(dry_run=False):
    """One pass over everything waiting. Returns rows sent."""
    state = load_state()
    sent = 0
    for name, kind in sources():
        path = os.path.join(DATA_DIR, name)
        offset = state.get(name, 0)
        # A file that shrank was rotated or replaced, so start it again.
        try:
            if os.path.getsize(path) < offset:
                log("%s is kleiner geworden, opnieuw vanaf het begin" % name)
                offset = 0
        except OSError:
            continue

        while True:
            lines, new_offset = read_new(path, offset)
            if not lines:
                break
            batch, rest = lines[:MAX_ROWS], len(lines) - MAX_ROWS
            if rest > 0:
                # Only claim the bytes this batch actually covers.
                new_offset = offset + len(("\n".join(batch) + "\n").encode())
            rows = power_rows(batch) if kind == "power" else question_rows(batch)
            if rows and not post(kind, rows, dry_run):
                log("%s: gestopt, mark blijft op %d" % (name, offset))
                save_state(state)
                return sent
            sent += len(rows)
            offset = new_offset
            state[name] = offset
            if not dry_run:
                save_state(state)
            if rest <= 0:
                break
    if sent:
        log("%d regels verstuurd" % sent)
    return sent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true", help="één keer proberen en stoppen")
    ap.add_argument("--dry-run", action="store_true", help="laten zien, niet sturen")
    ap.add_argument("--status", action="store_true", help="tonen wat er klaarstaat")
    args = ap.parse_args()

    if args.status:
        state = load_state()
        total = 0
        for name, _ in sources():
            path = os.path.join(DATA_DIR, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            waiting = max(0, size - state.get(name, 0))
            total += waiting
            print("%-28s %8.1f kB klaar van %8.1f kB"
                  % (name, waiting / 1024, size / 1024))
        print("-" * 56)
        print("%-28s %8.1f kB wacht op verzending" % ("totaal", total / 1024))
        return

    if not URL and not args.dry_run:
        print("SYNC_URL is niet gezet, dus er is nog nergens om heen te sturen.",
              file=sys.stderr)
        print("Draai met --dry-run om te zien wat er klaarstaat.", file=sys.stderr)
        sys.exit(2)

    if args.once or args.dry_run:
        run_once(args.dry_run)
        return

    log("elke %d seconden proberen, doel: %s" % (EVERY, URL))
    while True:
        try:
            run_once()
        except Exception as e:
            log("onverwachte fout: %r" % e)
        time.sleep(EVERY)


if __name__ == "__main__":
    main()
