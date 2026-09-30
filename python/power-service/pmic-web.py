#!/usr/bin/env python3
"""
pmic-web.py

Live power meter for the Raspberry Pi 5, served as a small web page.

The Pi 5's PMIC reports per-rail voltage and current, so the board can measure
its own consumption with no extra hardware. This samples those rails in the
background and serves a page showing watts over time, plus a box to send a
question to Ollama and get the energy cost of that single answer.

It measures the rails INSIDE the Pi. It does not include the Whisplay HAT's
screen and speaker, anything on USB, or the PiSugar's conversion losses, so it
reads lower than what the battery actually gives up. For "what did the thinking
cost", which happens on VDD_CORE, that is the right boundary.

Run:  python3 pmic-web.py
Then open http://<pi-address>:8425/
"""

import csv
import io
import json
import os

try:
    import fcntl          # Linux only; without it the panel sensor is skipped
except ImportError:
    fcntl = None
import re
import socket
import subprocess
import threading
import time
import html
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Everything below can be overridden with an environment variable, so this
# runs on a bot that does not live in /home/pi.
def env(name, default):
    return os.environ.get(name, default)


PORT = int(env("PMIC_WEB_PORT", "8425"))
SAMPLE_HZ = int(env("PMIC_SAMPLE_HZ", "10"))
HISTORY_SECONDS = int(env("PMIC_HISTORY_SECONDS", "300"))
OLLAMA = env("OLLAMA_ENDPOINT", "http://localhost:11434")
BOT_DIR = env("SOLARBOT_DIR", "/home/pi/solarbot")
ENV_FILE = env("SOLARBOT_ENV_FILE", os.path.join(BOT_DIR, ".env"))
LOG_FILE = env("SOLARBOT_LOG_FILE", os.path.join(BOT_DIR, "chatbot.log"))

# Everything measured is written here so a restart does not lose it.
#   questions.jsonl      one JSON object per question, append only
#   power-YYYY-MM-DD.csv one row per second: epoch, total watts, cpu watts
DATA_DIR = env("POWER_DATA_DIR", os.path.join(BOT_DIR, "data", "power"))
QUESTIONS_FILE = os.path.join(DATA_DIR, "questions.jsonl")

# PiSugar's own server. It reports charge level and terminal voltage but not
# current: on this board the output-current registers read zero, which is why
# the meter reads the Pi's PMIC instead.
# The INA219 on the charge input, between the solar panel and its converter.
# Optional: if it is not wired up, everything else carries on without it.
I2C_BUS = env("I2C_BUS", "/dev/i2c-1")
INA219_ADDR = int(env("INA219_ADDR", "0x40"), 16)
INA219_SHUNT_OHMS = float(env("INA219_SHUNT_OHMS", "0.1"))

PISUGAR = (env("PISUGAR_HOST", "127.0.0.1"), int(env("PISUGAR_PORT", "8423")))
BATTERY_MAH = int(env("BATTERY_MAH", "1200"))
BATTERY_NOMINAL_V = float(env("BATTERY_NOMINAL_V", "3.7"))

# The cloud comparison.
#
# ecocost turns a model name and a token count into an estimate of what that
# question would have cost in a data centre. It cannot measure anything, it
# cannot be told what was measured here, and it has never heard of the half
# billion parameter model this bot runs. So it is only ever used the other way
# round: to price the same question as if it had been asked of a cloud model,
# next to the joules that were actually spent answering it here.
#
# Everything it needs ships inside the package, so it keeps working with no
# network, which on this Pi is the normal state rather than the exception.
try:
    import ecocost
except Exception:                                    # not installed, or broken
    ecocost = None

COMPARE_MODEL = env("COMPARE_MODEL", "gpt-4o")
# The question log records how long the answering phase lasted but not how many
# tokens came out of it, so the count is derived from a rate. Measured on this
# Pi through /api/ask: qwen2.5:0.5b produces about 13 tokens a second. Measure
# it again with the ask panel if the model changes, and set the variable.
TOKENS_PER_SECOND = float(env("TOKENS_PER_SECOND", "13.3"))
# Rough across English prose, and only ever used for the question and the
# system prompt, which are both short enough that the error stays small next to
# the factor-of-eighteen uncertainty in the estimate itself.
CHARS_PER_TOKEN = 4.0
BATTERY_JOULES = BATTERY_MAH / 1000 * 3600 * BATTERY_NOMINAL_V   # about 15980 J

# How far the battery must fall before a calibration figure means anything.
# The reading has been seen to swing 4 to 5 points in two minutes purely from
# load, so a small drop is noise, not discharge.
MIN_DROP_PERCENT = 5.0

_ADC = re.compile(r"^\s*(\S+?)_(A|V)\s+(?:current|volt)\(\d+\)=([0-9.]+)")
_STATE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*switch to:\s*(\w+)")


def read_env():
    """Pull OLLAMA_MODEL and SYSTEM_PROMPT out of the chatbot's .env.

    So the page can default to the same model the bot uses, and optionally send
    the same system prompt. Comparing a bare prompt against the bot's prompt is
    comparing two different amounts of work.
    """
    model, system, key = "qwen2.5:0.5b", "", None
    try:
        lines = open(ENV_FILE, encoding="utf-8", errors="replace").read().splitlines()
    except Exception:
        return model, system
    buf = []
    for line in lines:
        if key == "SYSTEM_PROMPT":                # may span several lines
            buf.append(line)
            if line.rstrip().endswith('"'):
                system = "\n".join(buf).rstrip().rstrip('"')
                key = None
            continue
        s = line.strip()
        if s.startswith("OLLAMA_MODEL="):
            model = s.split("=", 1)[1].strip().strip('"').strip("'")
        elif s.startswith("SYSTEM_PROMPT="):
            v = s.split("=", 1)[1].lstrip()
            if v.startswith('"') and not v.rstrip().endswith('"'):
                key, buf = "SYSTEM_PROMPT", [v[1:]]
            else:
                system = v.strip().strip('"').strip("'")
    return model, system


BOT_MODEL, BOT_SYSTEM = read_env()

samples = deque(maxlen=SAMPLE_HZ * HISTORY_SECONDS)
samples_lock = threading.Lock()
idle_watts = 0.0

states = deque(maxlen=800)
states_lock = threading.Lock()

questions = deque(maxlen=200)
questions_lock = threading.Lock()

battery = {"level": None, "volts": None, "plugged": None, "charging": None,
           "ok": False, "at": 0}

panel = {"volts": None, "amps": None, "watts": None, "ok": False}
panel_lock = threading.Lock()
joules_in = 0.0          # everything the panel has delivered since start
battery_lock = threading.Lock()

# Running total of every joule the meter has counted since this process began,
# so a calibration session can compare it against what the battery actually lost.
joules_total = 0.0
session = {"t": 0.0, "joules": 0.0, "level": None}
session_lock = threading.Lock()

_RECOG = re.compile(r"Audio recognized:\s*(.+?)\s*$")


def read_power():
    """Return (total_watts, core_watts) from one PMIC read."""
    try:
        out = subprocess.run(["vcgencmd", "pmic_read_adc"],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return 0.0, 0.0
    amps, volts = {}, {}
    for line in out.splitlines():
        m = _ADC.match(line)
        if m:
            rail, kind, val = m.group(1), m.group(2), float(m.group(3))
            (amps if kind == "A" else volts)[rail] = val
    total = sum(a * volts.get(r, 0.0) for r, a in amps.items())
    core = amps.get("VDD_CORE", 0.0) * volts.get("VDD_CORE", 0.0)
    return total, core


def read_panel():
    """Read the INA219 over raw I2C. Returns (volts, amps, watts) or None.

    Two plain register reads, no library: the bus voltage register holds the
    reading in its top 13 bits at 4 mV a step, and the shunt register is a
    signed value at 10 uV a step, which over a known shunt gives the current.
    """
    if fcntl is None:
        return None
    try:
        fd = os.open(I2C_BUS, os.O_RDWR)
    except Exception:
        return None
    try:
        fcntl.ioctl(fd, 0x0703, INA219_ADDR)      # I2C_SLAVE

        def reg(r):
            os.write(fd, bytes([r]))
            b = os.read(fd, 2)
            return (b[0] << 8) | b[1]

        volts = (reg(0x02) >> 3) * 0.004
        raw = reg(0x01)
        if raw > 32767:
            raw -= 65536
        amps = raw * 1e-5 / INA219_SHUNT_OHMS
        return volts, amps, volts * amps
    except Exception:
        return None
    finally:
        try:
            os.close(fd)
        except Exception:
            pass


def pisugar(cmd):
    """One request to the PiSugar server. Returns the value after the colon."""
    try:
        s = socket.create_connection(PISUGAR, timeout=3)
        try:
            s.sendall((cmd + "\n").encode())
            reply = s.recv(256).decode("utf-8", "replace").strip()
        finally:
            s.close()
    except Exception:
        return None
    if ":" not in reply:
        return None
    val = reply.split(":", 1)[1].strip()
    return None if "not connected" in val.lower() else val


def battery_poller():
    """Follow the PiSugar. It updates about once a second, so 5 s is plenty."""
    while True:
        level = pisugar("get battery")
        volts = pisugar("get battery_v")
        plug = pisugar("get battery_power_plugged")
        chrg = pisugar("get battery_charging")
        with battery_lock:
            try:
                battery["level"] = round(float(level), 2) if level else None
                battery["volts"] = round(float(volts), 3) if volts else None
            except ValueError:
                battery["level"] = battery["volts"] = None
            battery["plugged"] = (plug == "true") if plug else None
            battery["charging"] = (chrg == "true") if chrg else None
            battery["ok"] = battery["level"] is not None
            battery["at"] = time.time()
        # Anchor a session the first time we get a reading.
        with session_lock:
            if session["level"] is None and battery["ok"]:
                session.update({"t": time.time(), "joules": joules_total,
                                "level": battery["level"]})
        time.sleep(5)


def session_report():
    with battery_lock:
        lvl, plugged = battery["level"], battery["plugged"]
    with session_lock:
        start_t, start_j, start_lvl = session["t"], session["joules"], session["level"]
    measured = max(0.0, joules_total - start_j)
    out = {"seconds": round(time.time() - start_t, 1) if start_t else 0,
           "measured_joules": round(measured, 1),
           "start_level": start_lvl, "level": lvl,
           "plugged": plugged, "drop": None,
           "battery_joules": None, "factor": None}
    out["min_drop"] = MIN_DROP_PERCENT
    out["suspect"] = False
    if lvl is not None and start_lvl is not None:
        drop = start_lvl - lvl
        out["drop"] = round(drop, 3)
        # The PiSugar derives its percentage from voltage, and voltage sags
        # under load, so the reading swings by several points on its own.
        # Anything smaller than MIN_DROP_PERCENT is inside that noise.
        if drop >= MIN_DROP_PERCENT and measured > 0 and plugged is False:
            real = drop / 100.0 * BATTERY_JOULES
            factor = real / measured
            out["battery_joules"] = round(real, 1)
            # Below 1 is physically impossible: the battery also pays for the
            # HAT and the conversion losses, so it must give up more than the
            # rails consume. A number under 1 means the battery reading is not
            # tracking energy over this window, not that the bot is efficient.
            if factor < 1.0:
                out["suspect"] = True
            else:
                out["factor"] = round(factor, 2)
    return out


CSV_HEADER = ("time,epoch,total_watts,cpu_watts,battery_percent,battery_volts,"
              "power_plugged,panel_volts,panel_amps,panel_watts")


def ensure_header(path, header):
    """Start the file with a header, and set aside any file written to an
    older layout rather than appending rows that no longer line up."""
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                if f.readline().strip() == header:
                    return
            os.rename(path, path + ".old-format")
        append_line(path, header)
    except Exception:
        pass


def append_line(path, line):
    """Append one line, never letting a disk problem take the meter down."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_questions():
    """Read back what earlier runs measured, so the page keeps its history."""
    try:
        with open(QUESTIONS_FILE, encoding="utf-8", errors="replace") as f:
            rows = f.readlines()[-questions.maxlen:]
    except Exception:
        return
    for line in rows:
        line = line.strip()
        if not line:
            continue
        try:
            questions.append(json.loads(line))
        except Exception:
            continue


def sampler():
    """Sample the PMIC forever, keep a rolling idle baseline, log 1 Hz to CSV."""
    global idle_watts, joules_total, joules_in
    dt = 1.0 / SAMPLE_HZ
    bucket, bucket_sec = [], 0
    last_t = None
    tick = 0
    while True:
        t = time.time()
        total, core = read_power()

        # The panel changes slowly and shares the bus with the PiSugar, so a
        # couple of reads a second is plenty.
        tick += 1
        if tick % max(1, SAMPLE_HZ // 2) == 0:
            p = read_panel()
            with panel_lock:
                if p is None:
                    panel.update(volts=None, amps=None, watts=None, ok=False)
                else:
                    panel.update(volts=round(p[0], 3), amps=round(p[1], 4),
                                 watts=round(p[2], 4), ok=True)
        with panel_lock:
            pw = panel["watts"]

        if last_t is not None:
            step = min(2.0, t - last_t)
            joules_total += total * step
            if pw:
                joules_in += pw * step
        last_t = t

        # One averaged row per second. Ten a second would be 860k rows a day.
        sec = int(t)
        if bucket_sec and sec != bucket_sec and bucket:
            n = len(bucket)
            path = os.path.join(DATA_DIR, "power-%s.csv"
                                % time.strftime("%Y-%m-%d", time.localtime(bucket_sec)))
            ensure_header(path, CSV_HEADER)
            with battery_lock:
                lvl, volts, plug = battery["level"], battery["volts"], battery["plugged"]
            with panel_lock:
                pv, pa, pwt = panel["volts"], panel["amps"], panel["watts"]
            append_line(path, "%s,%d,%.4f,%.4f,%s,%s,%s,%s,%s,%s" % (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(bucket_sec)),
                bucket_sec,
                sum(b[0] for b in bucket) / n,
                sum(b[1] for b in bucket) / n,
                "" if lvl is None else "%.2f" % lvl,
                "" if volts is None else "%.3f" % volts,
                "" if plug is None else ("yes" if plug else "no"),
                "" if pv is None else "%.3f" % pv,
                "" if pa is None else "%.4f" % pa,
                "" if pwt is None else "%.4f" % pwt))
            bucket = []
        bucket_sec = sec
        bucket.append((total, core))

        with samples_lock:
            samples.append((t, total, core, pw if pw is not None else 0.0))
            # Idle baseline: mean of the quietest tenth of the last two minutes.
            recent = sorted(s[1] for s in samples if t - s[0] < 120)
            if len(recent) > 20:
                k = max(1, len(recent) // 10)
                idle_watts = sum(recent[:k]) / k
        time.sleep(max(0.0, dt - (time.time() - t)))


def loaded_model():
    """Ask Ollama which model is resident right now.

    More honest than reading OLLAMA_MODEL from .env, because that only says what
    the bot was configured with, not what actually ran. Embedding models are
    filtered out; they are loaded for RAG alongside the chat model.
    """
    try:
        with urllib.request.urlopen(OLLAMA + "/api/ps", timeout=5) as r:
            names = [m.get("name", "") for m in json.loads(r.read()).get("models", [])]
        names = [n for n in names if "embed" not in n.lower()]
        if names:
            return names[0]
    except Exception:
        pass
    return BOT_MODEL


def close_question(cycle, t_end, text):
    """Turn one finished listen/asr/answer cycle into a row for the page."""
    t0 = cycle.get("listening") or cycle.get("asr") or cycle.get("answer")
    if t0 is None or "answer" not in cycle:
        return

    def phase(a, b):
        rows = window(a, b)
        j, jc = energy(rows)
        return {"seconds": round(max(0.0, b - a), 2),
                "joules": round(j, 1),
                "joules_core": round(jc, 1),
                "peak": round(max((r[1] for r in rows), default=0.0), 2)}

    # Only log what we actually measured. Phases are seeded from the tail of
    # the log at startup, so old cycles would otherwise show up with 0 J
    # because there is no power history covering them.
    if len(window(t0, t_end)) < 3:
        return

    t_asr = cycle.get("asr", cycle["answer"])
    t_ans = cycle["answer"]
    whole = phase(t0, t_end)
    base = idle_watts * max(0.0, t_end - t0)
    with battery_lock:
        plugged, blvl, bvolt = battery["plugged"], battery["level"], battery["volts"]
    row = {
        "t": round(t0, 1),
        "iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
        "text": (text or "").strip()[:200],
        "model": loaded_model(),
        "source": "mains" if plugged else ("battery" if plugged is False else "unknown"),
        "battery_percent": blvl,
        "battery_volts": bvolt,
        "total": whole,
        "above_idle": round(max(0.0, whole["joules"] - base), 1),
        "idle_watts": round(idle_watts, 3),
        "phases": {
            "listening": phase(t0, t_asr),
            "transcribing": phase(t_asr, t_ans),
            "answering": phase(t_ans, t_end),
        },
    }
    with questions_lock:
        questions.append(row)
    append_line(QUESTIONS_FILE, json.dumps(row))


def log_tailer():
    """Follow the chatbot's log: state transitions, and whole question cycles.

    The bot already prints "[2026-09-15 15:49:05] switch to: asr" on every
    transition and "Audio recognized: ..." once it has your words, so both the
    phase bands and a per-question log can be built without touching the bot.
    """
    pos = None
    first = True
    cycle = None
    last_text = ""
    while True:
        try:
            with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                f.seek(0, 2)
                size = f.tell()
                if first:
                    # Seed phases from the tail of the log, but start the
                    # question log empty: there is no power history for
                    # anything that happened before this process started.
                    pos = max(0, size - 400_000)
                    first = False
                elif pos is None or pos > size:   # log was rotated or truncated
                    pos = size
                f.seek(pos)
                for line in f:
                    r = _RECOG.search(line)
                    if r:
                        last_text = r.group(1)
                        continue
                    m = _STATE.search(line)
                    if not m:
                        continue
                    try:
                        t = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
                    except Exception:
                        continue
                    name = m.group(2)
                    with states_lock:
                        if not states or states[-1][1] != name:
                            states.append((t, name))

                    if cycle is not None and "answer" in cycle and name != "answer":
                        close_question(cycle, t, last_text)
                        cycle, last_text = None, ""
                    if name == "listening":
                        cycle = {"listening": t}
                    elif cycle is not None and name in ("asr", "answer"):
                        cycle.setdefault(name, t)
                    elif name == "sleep":
                        cycle = None
                pos = f.tell()
        except Exception:
            pass
        time.sleep(0.5)


def window(t0, t1):
    with samples_lock:
        return [s for s in samples if t0 <= s[0] <= t1]


def energy(rows):
    """Integrate watts over time -> joules. Returns (total_j, core_j)."""
    j_tot = j_core = 0.0
    for i in range(1, len(rows)):
        dt = rows[i][0] - rows[i - 1][0]
        j_tot += rows[i - 1][1] * dt
        j_core += rows[i - 1][2] * dt
    return j_tot, j_core


def ollama_models():
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=10) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:
        return []


def ollama_ask(model, prompt, system=""):
    """Try /api/chat first; models with no chat template fall back to generate."""
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    body = json.dumps({"model": model,
                       "messages": msgs,
                       "stream": False, "keep_alive": -1}).encode()
    req = urllib.request.Request(OLLAMA + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
        text = (d.get("message") or {}).get("content", "")
        if text.strip():
            return text, "chat", d.get("eval_count")
    except urllib.error.HTTPError:
        pass
    except Exception as e:
        return "(error: %s)" % e, "chat", None

    raw = (system + "\n\n" + prompt) if system else prompt
    body = json.dumps({"model": model, "prompt": raw,
                       "stream": False, "keep_alive": -1}).encode()
    req = urllib.request.Request(OLLAMA + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
        return d.get("response", ""), "generate", d.get("eval_count")
    except Exception as e:
        return "(error: %s)" % e, "generate", None


PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Solarbot power</title>
<style>
 :root{--bg:#fff5d1;--ink:#2b2320;--line:#c9b98a;--hi:#d2691e;--core:#3a7d6c;
   --dim:#9a8a6a;--wide:1100px}
 *{box-sizing:border-box}
 body{margin:0;padding:0 16px 28px;font:14px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--ink)}
 h1{font-size:22px;font-weight:600;margin:0 0 14px;text-align:center;letter-spacing:.02em}
 h2{font-size:15px;font-weight:600;margin:0 0 4px;letter-spacing:.01em}
 /* Keep the measure readable instead of stretching to the window. */
 header>h1,header>.row,header>.tabs,.panel{max-width:var(--wide);margin-left:auto;margin-right:auto}
 /* The stats stay put while you move between tabs: they are context for all of them. */
 header{position:sticky;top:0;z-index:5;background:var(--bg);padding:16px 0 0;
        box-shadow:0 10px 12px -10px rgba(43,35,32,.18)}
 .strip .card{flex:1 1 120px;padding:8px 12px}
 .strip .big{font-size:21px}
 .tabs{display:flex;gap:3px;border-bottom:1px solid var(--line);margin:4px 0 0}
 .tabs button{background:none;border:1px solid transparent;border-bottom:none;
   color:var(--dim);border-radius:9px 9px 0 0;padding:8px 18px;cursor:pointer;
   margin-bottom:-1px}
 .tabs button:hover{color:var(--ink)}
 .tabs button.on{background:#fffdf5;border-color:var(--line);color:var(--ink);font-weight:600}
 /* Left: what is happening now. Right: everything you look back at. */
 .tabs button.apart{margin-left:auto}
 /* Pushed right where there is room. A phone takes this back in one place,
    because an element that cannot shrink below its own words is exactly what
    gets shoved off the edge of a narrow screen. */
 .pushright{margin-left:auto}
 .panel{padding-top:20px}
 .panel[hidden]{display:none}
 /* Tooltips must never run off a narrow screen. */
 .i:hover::after,.i:focus::after{max-width:calc(100vw - 32px)}
 /* Anything table-shaped may scroll sideways rather than break the layout. */
 #qlog,#out,#reportbody,#scatter{overflow-x:auto}

 @media (max-width:700px){
   body{padding:0 12px 24px}
   /* A sticky header eats half a phone screen, so let it scroll away. */
   header{position:static;box-shadow:none;padding-top:12px}
   h1{font-size:19px;margin-bottom:10px}
   .strip .big{font-size:18px}
   .big .sub{font-size:12px;margin-left:5px}
   /* Five tabs will not fit side by side, so they scroll. The old
      -webkit-overflow-scrolling has done nothing since iOS 13 except promote
      its children to their own layers, which is half of why the selected tab
      lost its letters. */
   .tabs{overflow-x:auto}
   .tabs button{padding:8px 12px;font-size:13px;white-space:nowrap}
   .tabs button.apart{margin-left:12px}
   .panel{padding-top:16px}
   canvas{height:220px}
   .card{flex:1 1 calc(50% - 6px)}
   textarea{min-height:76px}
   table{font-size:12px}
   td.q{max-width:180px}
 }
 .btn{display:inline-block;background:var(--hi);color:#fff;border:1px solid var(--hi);
       border-radius:8px;padding:8px 16px;cursor:pointer;text-decoration:none;font-size:14px}
 .rp{background:#fffdf5;border:1px solid var(--line);border-radius:10px;
     padding:13px 16px;margin:10px 0}
 .rp .when{font-size:12px;opacity:.7;margin-right:8px}
 .rp .tag{display:inline-block;background:#f0e6c8;border-radius:20px;
          padding:1px 9px;margin-right:5px;font-size:12px}
 .rp blockquote{margin:9px 0 10px;font-size:16px;font-style:italic}
 .rp .cost{font-size:13px} .rp .cost b{font-size:20px;font-style:normal}
 .rp .sep{margin:0 7px;opacity:.4}
 .rp .bars{margin-top:9px;display:flex;gap:14px;flex-wrap:wrap;font-size:12px;opacity:.85}
 .row{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px}
 .card{background:#fffdf5;border:1px solid var(--line);border-radius:10px;padding:12px 14px;flex:1 1 150px}
 .big{font-size:30px;font-weight:600;line-height:1.1;font-variant-numeric:tabular-nums}
 /* A second reading alongside the main one, so every tile stays two lines tall. */
 .big .sub{font-size:14px;font-weight:400;opacity:.6;margin-left:8px}
 .lbl{font-size:11px;text-transform:uppercase;letter-spacing:.06em;opacity:.65;
      white-space:nowrap}
 canvas{width:100%;height:320px;display:block;background:#fffdf5;
        border:1px solid var(--line);border-radius:10px}
 textarea,select,button{font:inherit;border:1px solid var(--line);border-radius:8px;padding:8px;background:#fffdf5;color:inherit}
 textarea{width:100%;min-height:60px;resize:vertical}
 button{background:var(--hi);color:#fff;border-color:var(--hi);cursor:pointer;padding:8px 16px}
 button[disabled]{opacity:.5;cursor:default}
 table{border-collapse:collapse;width:100%;margin-top:18px;font-size:13px;
       font-variant-numeric:tabular-nums}
 th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}
 td.n,th.n{white-space:nowrap}
 /* Headings stay on one line, so the ? never drops below its label. */
 th{white-space:nowrap}
 /* The question itself may wrap, but not stretch the whole table. */
 td.q{max-width:260px;white-space:normal}
 /* One stacked bar instead of three columns of "x s / y J". */
 #qlog{overflow-x:auto;margin-top:14px}
 #scatter{margin-top:14px}
 .pbar{display:flex;height:11px;border-radius:3px;overflow:hidden;min-width:96px;
       background:#ece2c4;cursor:help}
 .pbar span{display:block;height:100%}
 th{font-size:11px;text-transform:uppercase;letter-spacing:.06em;opacity:.65}
 .ans{white-space:pre-wrap;background:#fffdf5;border:1px solid var(--line);border-radius:8px;padding:10px;margin-top:10px}
 .key{font-size:12px;opacity:.7;margin-top:6px}
 .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}
 details{margin-top:22px;background:#fffdf5;border:1px solid var(--line);border-radius:10px;padding:10px 14px}
 summary{cursor:pointer;font-weight:600;font-size:13px}
 details p{font-size:13px;margin:10px 0 0}
 details table{margin-top:10px}
 details td:first-child{white-space:nowrap;opacity:.75}
 .i{display:inline-block;width:15px;height:15px;text-align:center;
    border:1.5px solid var(--ink);color:var(--ink);background:transparent;
    border-radius:50%;font:700 10px/12px system-ui,sans-serif;
    opacity:.55;cursor:help;margin-left:5px;position:relative;vertical-align:middle}
 .i::before{content:'?'}
 .i:hover,.i:focus{opacity:1;outline:none;background:var(--ink);color:var(--bg)}
 .i:hover::after,.i:focus::after{content:attr(data-tip);position:absolute;left:50%;
    transform:translateX(-50%);bottom:150%;width:250px;background:#2b2320;color:#fff5d1;
    padding:9px 11px;border-radius:8px;z-index:20;text-align:left;white-space:pre-line;
    font:400 12px/1.5 system-ui,sans-serif;text-transform:none;letter-spacing:0;
    box-shadow:0 5px 18px rgba(0,0,0,.28)}
 .i.r:hover::after,.i.r:focus::after{left:auto;right:-4px;transform:none}
 /* Near the top of the page there is no room above, so open downwards. */
 .i.d:hover::after,.i.d:focus::after{bottom:auto;top:150%}
 /* Against the left edge, anchor left instead of centring. */
 .i.l:hover::after,.i.l:focus::after{left:-4px;right:auto;transform:none}
 /* A roomier one, for the note that explains the whole measurement. */
 .i.wide:hover::after,.i.wide:focus::after{width:430px}

 /* The bot's own face. These are the very drawings its little screen shows,
    so the page wears the same expression as the thing on the desk. */
 .mast{display:flex;align-items:center;justify-content:center;gap:14px;margin:0 0 14px}
 .mast h1{margin:0;text-align:left}
 .face{width:76px;height:76px;flex:0 0 auto;margin:0;cursor:pointer;
       display:flex;align-items:center;justify-content:center}
 .face img{width:100%;height:100%;object-fit:contain;display:block}
 .doing{font-size:11px;text-transform:uppercase;letter-spacing:.07em;opacity:.6;margin-top:3px}
 .doing.preview{opacity:.85;font-style:italic}
 @media (max-width:700px){.face{width:56px;height:56px}}

 /* ---------------------------------------------------------------------
    "paper": the skin taken straight from solararchive.cmama.xyz. Her site
    downloads no font at all, so prose falls back to the browser's serif and
    the machine readable parts are Courier. Headings are not bold, corners
    are never rounded (one border-radius in her whole stylesheet), and things
    are separated by rules rather than boxed in. Switch it at the bottom
    right; the choice is remembered. */
 html.paper body{font-family:Georgia,"Times New Roman",serif;color:#000}
 html.paper h1,html.paper h2{font-weight:normal}
 html.paper h1{text-indent:-.15rem}
 /* Courier for every number: her machine voice, and the digits keep the same
    width so a reading no longer jitters while it changes */
 html.paper .big,html.paper .lbl,html.paper .doing,html.paper .tag,
 html.paper .rp .when,html.paper table,html.paper #pagecost{
   font-family:"Courier New",Courier,monospace}
 html.paper .big{font-weight:normal}
 html.paper .lbl{letter-spacing:0}
 html.paper .card,html.paper .btn,html.paper button,html.paper canvas,
 html.paper textarea,html.paper select,html.paper .rp,html.paper .tag,
 html.paper .tabs button{border-radius:0}
 /* cards stop being little boxes and become ruled rows */
 html.paper .card{background:none;border:0;border-top:2px solid #000;
   padding:8px 12px 10px 0}
 html.paper .rp{background:none;border:0;border-top:1px solid #000}
 html.paper canvas{background:none;border:1px solid #000}
 html.paper button,html.paper .btn{background:none;color:#000;border:1px solid #000}
 html.paper button:hover,html.paper .btn:hover{background:#000;color:#fff5d1}
 html.paper .tabs{border-bottom:1px solid #000}
 html.paper .tabs button{color:#828282}
 html.paper .tabs button.on{background:none;border-color:transparent;
   border-bottom:2px solid #000;font-weight:normal;color:#000}
 html.paper a{text-decoration:none;border-bottom:1px solid;padding-bottom:.05em}
 html.paper a:hover{color:#828282}
 html.paper textarea,html.paper select{background:none;border:1px solid #000}
 html.paper .tag{background:none;border:1px solid #000}
 /* The last box on the page. Every card and report row had already turned into
    a rule with space under it, and this one white rounded panel was left over
    from before, sitting in the middle of the page shouting. Her site has no
    boxes at all, so it becomes a ruled section like the rest and the heading
    joins the machine voice. */
 html.paper details{background:none;border:0;border-top:1px solid #000;
   border-radius:0;padding:12px 0 0;margin-top:30px}
 /* The questions download sits right above this panel's rule but belongs to the
    table further up, so the gap a panel usually gets would open in the wrong
    place: between a line and the thing it is about. */
 #cmpbox{margin-top:10px}
 html.paper summary{font-family:"Courier New",Courier,monospace;
   font-weight:normal;text-transform:uppercase;letter-spacing:.09em;font-size:12px}
 html.paper summary::marker{color:#828282}
 html.paper details[open] summary{margin-bottom:2px}
 /* The answer itself was a second box inside the first one. */
 html.paper .ans{background:none;border:0;border-left:2px solid #000;
   border-radius:0;padding:2px 0 2px 14px;margin-top:14px;
   font-family:Georgia,"Times New Roman",serif}
 /* Her site states what the page weighed, down in the corner. This page is
    about what things cost, so it says the same thing in the same place. */
 #pagecost{display:none}
 html.paper #pagecost{display:block;position:fixed;left:7px;bottom:5px;
   font-size:11px;opacity:.75}
 #skin{position:fixed;right:7px;bottom:5px;font-size:11px;cursor:pointer;
   background:none;border:0;color:inherit;opacity:.55;padding:2px 4px;
   font-family:"Courier New",Courier,monospace;border-radius:0}
 #skin:hover{opacity:1}

 /* ---------------------------------------------------------------------
    Phone, last word.

    These belong at the end and not with the other narrow-screen rules higher
    up, because a media query adds no specificity: a plain rule further down
    the sheet beats it. Both of the things below were written up there first
    and quietly lost to the paper skin, which is declared after them. */
 @media (max-width:700px){
   /* Two across, without arithmetic. Sizing the cards at "half minus the gap"
      needs the gap to be what you think it is, and it was not: twelve, not
      eight, so two cards plus the gap came to four pixels more than the row
      and they fell back to one each. A grid is told the columns and works the
      gap out itself, so it cannot be wrong by four pixels. */
   .strip{display:grid;grid-template-columns:1fr 1fr;gap:8px}
   .strip .card{min-width:0;padding:7px 10px}
   .strip .lbl{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

   /* Five tabs wanted 356 pixels in a 333 pixel row, so the bar scrolled and
      Calibration hung off the end. Trimming the padding buys back 40 and they
      all fit. The overflow stays as a safety net for anything narrower again,
      but without a scrollbar drawn across the tabs. */
   /* Smaller again, to buy back the room the left/right split needs: without
      spare width margin-left:auto has nothing to push with. */
   .tabs button{padding:8px 6px;font-size:12px}
   .tabs button.apart{margin-left:auto}
   .tabs{scrollbar-width:none}
   .tabs::-webkit-scrollbar{display:none}

   /* The history row put the download link out to the right with margin-left
      auto. That is a nice touch on a wide screen and the thing most likely to
      be pushed off a narrow one, since the link cannot be made narrower than
      its own words. On a phone it just follows the day picker. */
   .pushright{margin-left:0}
   .key,#daynote,#dayfile{overflow-wrap:anywhere}

   /* A guard, not a fix: whatever else turns out to be a pixel too wide, it
      may not drag the whole page sideways with it. clip rather than hidden,
      because hidden would make the page a scroll container and break the
      boxes that are supposed to scroll on their own, and sticky with them. */
   html,body{overflow-x:clip}

   /* Fixed to the bottom corners is margin on a desktop and the last line of
      the text on a phone. Down here they simply end the page. */
   html.paper #pagecost,#skin{position:static;display:block;width:auto;
     text-align:center;margin:26px auto 0;opacity:.6}
 }
 /* A dropdown keeps the operating system's own look unless you take it off,
    which is why it stayed grey and rounded while everything around it changed. */
 html.paper select{appearance:none;-webkit-appearance:none;background:none;
   border:1px solid #000;border-radius:0;padding:7px 26px 7px 9px;
   font-family:"Courier New",Courier,monospace;color:#000;
   background-image:linear-gradient(45deg,transparent 50%,#000 50%),
                    linear-gradient(135deg,#000 50%,transparent 50%);
   background-position:right 13px center,right 8px center;
   background-size:5px 5px,5px 5px;background-repeat:no-repeat}
 html.paper select:focus{outline:1px solid #000;outline-offset:1px}
 /* The list that drops open is drawn separately from the button, so it stayed
    the browser's own white until told otherwise. */
 html.paper select option,html.paper select optgroup{background:#fff5d1;color:#000}
 html.paper select option:checked{background:#000;color:#fff5d1}
 html.paper .sw{border-radius:0}
 /* The bands on the live graph only appear once the bot has actually done
    something, so this states what the colours mean even when it is asleep. */
 /* The same stacked bar the report uses, blown up to the width of the graph.
    The bands on the graph only exist after the bot has answered something, so
    this doubles as the key for them while it is asleep. */
 .phasebar{margin-top:12px}
 .phasebar .pbar{height:16px;width:100%;min-width:0;border-radius:4px}
 html.paper .phasebar .pbar{border-radius:0;border:1px solid #000;background:none}
 /* The line under it is a plain .key, exactly like the one at the foot of the
    static table, so the same three colours are named the same way twice. */
</style>
<header>
<div class="mast">
  <figure class="face" id="face" tabindex="0"
    title="What the bot is doing right now, drawn the way its own screen draws it. Click to step through the states."><img id="faceimg" alt=""></figure>
  <div><h1>Solar bot</h1><div class="doing" id="doing">idle</div></div>
</div>
<div class="row strip">
  <div class="card"><div class="lbl">now<i class="i d l" tabindex="0" data-tip="Total power the Pi's internal rails are drawing at this moment. Power is a rate, like speed: it says how fast energy is being used, not how much in total."></i></div><div class="big"><span id="now">-</span> W</div></div>
  <div class="card"><div class="lbl">idle<i class="i d" tabindex="0" data-tip="What the Pi draws doing nothing: the quietest tenth of the last two minutes. Everything labelled 'cost of question' is measured against this baseline."></i></div><div class="big"><span id="idle">-</span> W</div></div>
  <div class="card"><div class="lbl">peak (5 min)<i class="i d" tabindex="0" data-tip="Highest total power seen in the window shown in the live graph."></i></div><div class="big"><span id="peak">-</span> W</div></div>
  <div class="card"><div class="lbl">cpu now<i class="i d" tabindex="0" data-tip="The VDD_CORE rail on its own, which is the processor. This is where the language model actually does its work, so it is the part that jumps when the bot thinks."></i></div><div class="big"><span id="core">-</span> W</div></div>
  <div class="card"><div class="lbl">panel in<i class="i d" tabindex="0" data-tip="What the solar panel is delivering right now, measured by the INA219 on the charge input. Compare it with 'now': if this is the smaller number, the bot is running the battery down even while it charges."></i></div>
    <div class="big"><span id="pw">-</span> W<span class="sub"><span id="pv">-</span> V</span></div></div>
  <div class="card"><div class="lbl">battery <span id="bstate" style="text-transform:none"></span><i class="i r d" tabindex="0" data-tip="Charge and terminal voltage as reported by the PiSugar. The percentage is derived from voltage, not from counting energy, so it dips when the bot works hard and recovers afterwards. Treat it as an indication, not a measurement."></i></div>
    <div class="big"><span id="blvl">-</span> %<span class="sub"><span id="bv">-</span> V</span></div></div>
</div>
<nav class="tabs">
  <button data-tab="live">Live</button>
  <button data-tab="static">Static</button>
  <button data-tab="history" class="apart">History</button>
  <button data-tab="report">Report</button>
  <button data-tab="calib">Calibration</button>
</nav>
</header>

<section id="tab-live" class="panel">
<div class="row" style="margin-bottom:10px;align-items:center;gap:8px">
  <div style="flex:0 0 auto"><span class="lbl">show energy as</span>
    <select id="unit" style="margin-left:6px">
      <option value="J">joules (J)</option>
      <option value="mWh">milliwatt-hours (mWh)</option>
      <option value="mAh">milliamp-hours (mAh)</option>
      <option value="pct">% of a full battery</option>
      <option value="sun">seconds of sunshine</option>
    </select></div>
  <div id="panelbox" style="flex:0 0 auto"><span class="lbl">solar panel</span><i class="i d" tabindex="0" data-tip="The rated power of your panel, in watts. Used only for the 'seconds of sunshine' unit: the energy of a question divided by this number.

Look on the back of the panel or on its packaging. If it lists volts and amps instead, multiply them: 6 V at 1 A is 6 W.

Note that the rating is for full sun, straight on. In practice you get less, so the seconds shown are a best case and the real time is longer. Measuring the panel input with a sensor would give the true figure."></i>
    <input id="panel" type="number" min="0.1" step="0.5" value="5" style="width:70px;margin-left:6px"> W</div>
  <div class="key" id="unitnote" style="margin:0"></div>
  <div class="pushright" style="flex:0 0 auto"><span class="lbl">what is measured</span><i class="i r d wide" tabindex="0" data-tip="Measured from the Raspberry Pi 5's own PMIC, which reports voltage and current for the board's internal rails. The Pi measures itself, no external sensor is involved.

What you see is the processor, the memory and the wifi chip. That is where the thinking happens. The green line is VDD_CORE, the processor on its own.

Not included: the Whisplay HAT with its screen backlight, speaker and microphone; anything on USB; the PiSugar stepping 3.7 V up to 5 V, which costs roughly 10 to 15 percent; and the Pi's own conversion down to each rail.

So the real drain on the battery is higher than the joules shown here, by an amount that has not been measured. Figures given as a percentage of a full battery come from those same joules, so they are a lower bound too.

To get the true battery cost, either put a sensor in the battery lead, or run a calibration on the Calibration tab."></i></div>
</div>
<canvas id="c" width="1200" height="440"></canvas>
<div class="phasebar" id="phasebar"></div>
<div class="key"><span class="sw" style="background:#d2691e"></span>total used
  <span class="sw" style="background:#3a7d6c;margin-left:12px"></span>cpu only (VDD_CORE)
  <span class="sw" style="background:#c9a227;margin-left:12px"></span>coming in from the panel</div>

<details id="askbox">
<summary>Ask a question from here</summary>
<div class="key">Sends one question straight to Ollama and measures just that answer. Handy for putting two
  models side by side without holding the button.</div>
<div class="row" style="margin-top:10px;align-items:flex-start">
  <div style="flex:1 1 420px">
    <div class="lbl" style="margin-bottom:4px">question</div>
    <textarea id="q">What is solar power, in two sentences?</textarea>
  </div>
  <div style="flex:0 1 240px">
    <div class="lbl" style="margin-bottom:4px">model</div>
    <select id="m" style="width:100%"></select>
  </div>
  <div style="flex:0 0 auto">
    <div class="lbl" style="margin-bottom:4px">&nbsp;</div>
    <button id="go">Ask &amp; measure</button>
  </div>
</div>
<div class="key"><label><input type="checkbox" id="sys" checked> use the same system prompt as the bot
  <span id="syslen"></span></label></div>
<div id="out"></div>
</details>

</section>

<section id="tab-history" class="panel" hidden>
<div class="row" style="align-items:center;gap:10px;margin-bottom:10px">
  <label>Day <select id="day"></select></label>
  <span class="key" id="daynote"></span>
  <span class="key pushright" id="dayfile"></span>
</div>
<canvas id="hc" width="1200" height="380"></canvas>
<div class="key">A whole day at once, so you can compare one day with another. Each
 minute of the log becomes one point, and the three lines are:
 <b>peak</b>, the busiest single second in that minute, faint and behind the rest;
 <b>total</b>, the average of everything the Pi drew that minute;
 <b>cpu</b>, the part of that average spent on the processor alone, in blue.
 The gap between total and cpu is everything that is not the processor: the
 screen, the microphone, the board itself.
 Gaps in the lines are left empty on purpose \u2014 that is the bot switched off,
 and a day with holes in it is worth being able to see.</div>
<div class="row" style="margin-top:12px">
  <div class="card"><div class="lbl">logged</div><div class="big"><span id="hcov">-</span></div></div>
  <div class="card"><div class="lbl">energy that day</div><div class="big"><span id="hwh">-</span> Wh</div></div>
  <div class="card"><div class="lbl">average</div><div class="big"><span id="hmean">-</span> W</div></div>
  <div class="card"><div class="lbl">peak</div><div class="big"><span id="hpeak">-</span> W</div></div>
  <div class="card"><div class="lbl">battery</div><div class="big"><span id="hbat">-</span></div></div>
</div>
</section>

<section id="tab-static" class="panel" hidden>
<div class="row" style="align-items:center;gap:10px;margin-bottom:14px">
  <label>Day <select id="qday"></select></label>
  <span class="key" id="qdaynote"></span>
</div>
<h2>Cost against duration</h2>
<div class="key">One dot per question: how long it took against what it cost. The numbers match the list
  below. Hover a dot to see the question.</div>
<div id="scatter"></div>

<h2 style="margin-top:26px">Questions asked on the device</h2>
<div class="key">Every time you hold the button on the bot, the whole cycle is logged here with the model
  that was loaded when it answered. Saved to disk, so restarts do not lose it.</div>
<div id="qlog"></div>
<div class="key" id="qfiles"></div>

<details id="cmpbox">
<summary>What if you had asked a cloud model?</summary>
<div class="key" id="comparenote"></div>
<div class="row" style="align-items:center;gap:10px;margin:10px 0 4px">
  <label>Compare with <select id="cmodel"></select></label>
</div>
<div id="compare"></div>
</details>
</section>

<section id="tab-calib" class="panel" hidden>
<h2>Battery calibration</h2>
<div class="key">The meter reads the Pi's internal rails, so it under-reports what the battery actually gives up.
  Unplug the power, reset below, ask a few dozen questions, and the charge the PiSugar loses against the
  joules counted here gives the correction factor.</div>
<div class="row" style="margin-top:12px">
  <div class="card"><div class="lbl">session length<i class="i d l" tabindex="0" data-tip="Time since you last pressed Reset session."></i></div><div class="big"><span id="slen">-</span></div></div>
  <div class="card"><div class="lbl">measured<i class="i d" tabindex="0" data-tip="Every joule the meter has counted since the reset, from the Pi's internal rails only."></i></div><div class="big"><span id="sj">-</span></div></div>
  <div class="card"><div class="lbl">battery drop<i class="i d" tabindex="0" data-tip="Percentage points the PiSugar has fallen since the reset. Because that percentage comes from voltage, it wanders on its own, which is why a calibration needs a large drop before it means anything."></i></div><div class="big"><span id="sdrop">-</span> %</div></div>
  <div class="card"><div class="lbl">energy from battery<i class="i d" tabindex="0" data-tip="That percentage drop turned into joules, assuming a full pack of 1200 mAh at 3.7 V, which is about 15980 J. This is what the battery is estimated to have actually handed over."></i></div><div class="big"><span id="sbj">-</span></div></div>
  <div class="card"><div class="lbl">correction factor<i class="i r d" tabindex="0" data-tip="Energy from battery divided by measured: how many joules the battery gives up for each joule the rails use. It has to be above 1, because the battery also pays for the screen, the speaker and the conversion losses. A result below 1 is shown as a warning instead, since it cannot be real."></i></div><div class="big"><span id="sfac">-</span></div></div>
</div>
<div class="key" id="snote"></div>
<div style="margin-top:16px"><button id="reset">Reset session</button></div>
</section>

<section id="tab-report" class="panel" hidden>
<div class="row" style="align-items:center;margin-bottom:6px">
  <h2 style="margin:0">Report</h2>
  <label style="margin-left:18px">Day <select id="rday"></select></label>
  <div class="pushright"><a id="dl" class="btn" download="solarbot-energy-report.html">Download</a></div>
</div>
<div id="reportbody"></div>
</section>

<div id="pagecost"></div>
<button id="skin" title="Switch between the original look and the one taken from the Sun Shines Bright site"></button>

<script>
const $=id=>document.getElementById(id);
let data=[],phases=[],BAT={j:15984,mah:1200,v:3.7};

// Every energy number the server sends is in joules. These are all just
// rescalings of the same measurement, for whoever is reading the page.
const UNITS={
  J  :{lbl:'J',   d:1, f:j=>j},
  mWh:{lbl:'mWh', d:2, f:j=>j/3.6},
  mAh:{lbl:'mAh', d:3, f:j=>j/BAT.v/3.6},
  pct:{lbl:'%',   d:3, f:j=>j/BAT.j*100},
  sun:{lbl:'s',   d:1, f:j=>j/Math.max(0.1,parseFloat($('panel').value)||5)}
};
const U=()=>UNITS[$('unit').value]||UNITS.J;
const fmtE=j=>{if(j==null||isNaN(j))return '-';const u=U();return u.f(j).toFixed(u.d)+' '+u.lbl;};
function unitNote(){
  const v=$('unit').value;
  // The panel rating only feeds the sunshine unit, so it only appears there.
  $('panelbox').style.display = (v==='sun') ? '' : 'none';
  $('unitnote').textContent =
    v==='J'  ?'1 joule is 1 watt for 1 second.':
    v==='mWh'?'What batteries and electricity bills use. 1 Wh = 3600 J.':
    v==='mAh'?'Phone-battery units, at '+BAT.v+' V nominal. The pack holds '+BAT.mah+' mAh.':
    v==='pct'?'Share of one full charge of the '+BAT.mah+' mAh pack ('+Math.round(BAT.j)+' J).':
              'How long the panel would need to make it, at its rated output.';
}
/* f: which of the five drawings belongs to this flow. The log calls the flows
   by their own names, while the drawings are named after what the bot puts on
   its screen, so the two vocabularies have to be tied together here. */
/* Every phase colour in one place, so the bands on the graph, the bar under it
   and the bars in the report can never drift apart again. Lighter than the old
   set: at full strength those read as heavy blocks on cream rather than as a
   key you can glance at. The paper column keeps the hues of her site but tinted
   towards the background instead of sitting at full saturation. */
const PHASE={
  listening   :{c:'#7ecb92', p:'#5fbda6'},
  detecting   :{c:'#b6e4c2', p:'#a6d8cb'},
  transcribing:{c:'#84acea', p:'#8b95de'},
  answering   :{c:'#f2ad72', p:'#f0897d'},
  image       :{c:'#cba6e0', p:'#c2c2c2'}
};
const PHC=n=>((PHASE[n]||{})[isPaper()?'p':'c'])||'#cccccc';
/* The same colour again, see-through, for the band painted behind the curve. */
const PHBAND=(n,a)=>{
  const h=PHC(n);
  return 'rgba('+parseInt(h.slice(1,3),16)+','+parseInt(h.slice(3,5),16)+','+
         parseInt(h.slice(5,7),16)+','+a+')';
};
const PH={
  listening     :{n:'listening',    c:'rgba(74,160,90,.20)', t:'#2f6b3c', f:'listening',
                  pc:'rgba(0,105,81,.16)',   pt:'#006951'},
  wake_listening:{n:'listening',    c:'rgba(74,160,90,.20)', t:'#2f6b3c', f:'detecting',
                  pc:'rgba(0,105,81,.16)',   pt:'#006951'},
  detecting     :{n:'detecting',    c:'rgba(74,160,90,.10)', t:'#2f6b3c', f:'detecting',
                  pc:'rgba(0,105,81,.09)',   pt:'#006951'},
  asr           :{n:'transcribing', c:'rgba(62,110,190,.20)',t:'#2b4c86', f:'recognizing',
                  pc:'rgba(22,45,171,.16)',  pt:'#162dab'},
  answer        :{n:'answering',    c:'rgba(210,105,30,.24)',t:'#9c4c14', f:'answering',
                  pc:'rgba(209,19,5,.16)',   pt:'#d11305'},
  image         :{n:'image',        c:'rgba(150,90,180,.18)',t:'#6b3f80', f:'answering',
                  pc:'rgba(130,130,130,.16)',pt:'#828282'},
  sleep         :{n:'',             c:null,                  t:'',        f:'idle',
                  pc:null,                   pt:''}
};

/* ---- the face ---------------------------------------------------------
   Five states, three frames each. The frames ping-pong (0,1,2,1) so the
   movement breathes instead of snapping back to the start. Clicking steps
   through the states by hand, which is the only way to see them while the
   chatbot itself is not running. */
const FACES=['idle','listening','detecting','recognizing','answering'];
const FSEQ=[0,1,2,1];
let faceState='idle', faceStep=0, facePreview=null;
FACES.forEach(s=>[0,1,2].forEach(i=>{const im=new Image();im.src='faces/'+s+'/'+i+'.png';}));
function paintFace(){
  const s=facePreview||faceState;
  $('faceimg').src='faces/'+s+'/'+FSEQ[faceStep%FSEQ.length]+'.png';
  $('doing').textContent=s+(facePreview?' · preview':'');
  $('doing').classList.toggle('preview',!!facePreview);
}
function stepFace(){
  const i=FACES.indexOf(facePreview||faceState);
  facePreview=FACES[(i+1)%FACES.length];
  if(facePreview===faceState) facePreview=null;
  faceStep=0; paintFace();
}
function setFace(flow){
  const s=(PH[flow]&&PH[flow].f)||'idle';
  if(s===faceState) return;
  faceState=s; faceStep=0; facePreview=null; paintFace();
}
async function poll(){
  try{
    const r=await fetch('api/samples');const j=await r.json();
    data=j.samples;phases=j.states||[];
    setFace(phases.length?phases[phases.length-1][1]:'sleep');
    const last=data[data.length-1];
    if(last){$('now').textContent=last[1].toFixed(2);$('core').textContent=last[2].toFixed(2);}
    $('idle').textContent=j.idle.toFixed(2);
    $('peak').textContent=(data.reduce((a,s)=>Math.max(a,s[1]),0)).toFixed(2);
    const p=j.panel||{};
    $('pw').textContent=p.watts==null?'-':p.watts.toFixed(p.watts<1?3:2);
    $('pv').textContent=p.volts==null?'-':p.volts.toFixed(2);
    const b=j.battery||{};
    $('blvl').textContent=b.level==null?'-':b.level.toFixed(1);
    $('bv').textContent=b.volts==null?'-':b.volts.toFixed(3);
    $('bstate').textContent=!b.ok?'(no PiSugar)':(b.charging?'· charging':(b.plugged?'· plugged in':'· on battery'));
    showSession(j.session||{});
    draw();
  }catch(e){}
}
function showSession(s){
  const hms=x=>{x=Math.round(x);const h=Math.floor(x/3600),m=Math.floor(x%3600/60);
    return h?h+'h '+m+'m':(m?m+'m '+(x%60)+'s':x+'s');};
  $('slen').textContent=s.seconds?hms(s.seconds):'-';
  $('sj').textContent=s.measured_joules==null?'-':fmtE(s.measured_joules);
  $('sdrop').textContent=s.drop==null?'-':s.drop.toFixed(2);
  $('sbj').textContent=s.battery_joules==null?'-':fmtE(s.battery_joules);
  $('sfac').textContent=s.factor==null?'-':('x'+s.factor.toFixed(2));
  const min=s.min_drop||5;
  let n='';
  if(s.plugged===true) n='Plugged in, so the battery is not discharging. Unplug to run a calibration.';
  else if(s.drop==null) n='Waiting for a battery reading.';
  else if(s.drop<min) n='Battery has dropped '+s.drop.toFixed(2)+' % so far. A calibration needs at least '+
    min+' %, because the PiSugar reads charge from voltage and voltage sags under load, which moves the '+
    'number by several points on its own. Expect this to take a while.';
  else if(s.suspect) n='That works out below 1, which cannot be right: the battery also pays for the screen, '+
    'the speaker and the conversion losses, so it must give up more than the rails use. The percentage is '+
    'not tracking energy over this window. A lithium cell holds about 3.8 V across most of its range, so in '+
    'the middle the reading barely moves. Run a longer and deeper discharge.';
  else if(s.factor!=null) n='Every joule this meter reports is about '+s.factor.toFixed(2)+
    ' joules out of the battery. So a 10 J answer really costs roughly '+(10*s.factor).toFixed(0)+' J of charge.';
  $('snote').textContent=n;
}
$('reset').onclick=async()=>{
  try{const r=await fetch('api/session/reset',{method:'POST'});showSession(await r.json());}catch(e){}
};
function draw(){
  const c=$('c'),x=c.getContext('2d');
  // Match the backing store to the size the canvas is actually shown at,
  // otherwise the drawing gets stretched to fit and everything looks squashed.
  const dpr=window.devicePixelRatio||1;
  const W=c.clientWidth||900,H=c.clientHeight||300;
  if(c.width!==Math.round(W*dpr)||c.height!==Math.round(H*dpr)){
    c.width=Math.round(W*dpr);c.height=Math.round(H*dpr);
  }
  x.setTransform(dpr,0,0,dpr,0,0);   // from here on, work in CSS pixels
  x.clearRect(0,0,W,H);
  if(data.length<2)return;
  const t0=data[0][0],t1=data[data.length-1][0],span=Math.max(1,t1-t0);
  const peak=Math.max(10,data.reduce((a,s)=>Math.max(a,s[1],s[3]||0),0)*1.15);
  const B=30;   // room along the bottom for clock labels
  const px=t=>(t-t0)/span*W, py=w=>H-B-(w/peak)*(H-B);
  const paper=document.documentElement.classList.contains('paper');
  for(let i=0;i<phases.length;i++){
    const p=PH[phases[i][1]];if(!p||!p.n)continue;
    const a=Math.max(0,px(phases[i][0]));
    const b=i+1<phases.length?px(phases[i+1][0]):W;
    if(b<=a)continue;
    /* detecting is the quiet sibling of listening and shares its colour, so it
       is painted fainter to stay apart from it. */
    x.fillStyle=PHBAND(p.n,p.n==='detecting'?0.17:0.32);x.fillRect(a,0,b-a,H);
    if(b-a>62&&p.n){x.fillStyle=paper?p.pt:p.t;
      x.font=paper?'12px "Courier New",Courier,monospace':'600 12px system-ui';
      x.fillText(p.n,a+6,17);}
  }
  x.strokeStyle='#e0d5b0';x.lineWidth=1;x.font='11px system-ui';x.fillStyle='#9a8a6a';
  const step=peak>24?4:2;
  for(let w=0;w<=peak;w+=step){x.beginPath();x.moveTo(0,py(w));x.lineTo(W,py(w));x.stroke();x.fillText(w+' W',6,py(w)-3);}
  // Clock along the bottom, so you can tell how much time the window covers.
  x.strokeStyle='#c9b98a';x.beginPath();x.moveTo(0,H-B);x.lineTo(W,H-B);x.stroke();
  x.textAlign='center';
  for(let i=0;i<=4;i++){
    const t=t0+span*i/4,xp=Math.min(W-30,Math.max(30,px(t)));
    x.fillText(new Date(t*1000).toLocaleTimeString(),xp,H-8);
  }
  x.textAlign='left';
  const line=(idx,col)=>{x.beginPath();x.strokeStyle=col;x.lineWidth=2;
    data.forEach((s,i)=>i?x.lineTo(px(s[0]),py(s[idx])):x.moveTo(px(s[0]),py(s[idx])));x.stroke();};
  line(2,'#3a7d6c');line(3,'#c9a227');line(1,'#d2691e');
}
async function models(){
  try{
    const r=await fetch('api/models');const j=await r.json();
    if(j.battery_joules){BAT={j:j.battery_joules,mah:j.battery_mah,v:j.battery_volts};}
    unitNote();
    const bot=j.bot_model||'';
    $('m').innerHTML=j.models.map(n=>{
      const isBot=(n===bot||n===bot+':latest'||n.replace(/:latest$/,'')===bot);
      return '<option value="'+n+'"'+(isBot?' selected':'')+'>'+n+(isBot?'  (the bot uses this)':'')+'</option>';
    }).join('');
    $('syslen').textContent=j.system_chars?'('+j.system_chars+' chars from .env)':'(none found in .env)';
    if(!j.system_chars){$('sys').checked=false;$('sys').disabled=true;}
  }catch(e){}
}
const esc=s=>(s||'').replace(/[<>&]/g,ch=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[ch]));

// Column explanations, so the page can be read without anyone explaining it.
const TIP={
 model:'Which model was loaded in Ollama when this answer was produced, asked of Ollama itself rather than read from the config.',
 source:'Whether the bot was on mains power or running from the battery when this question was asked.',
 batt:'Battery level at the moment the question started.',
 dur:'How long the whole cycle took, from pressing the button to the answer being finished.',
 total:'All the energy that flowed during the question, including the part the bot would have used anyway just by being switched on.',
 cost:'The extra energy this question added on top of simply being switched on: total energy minus idle power times duration. This is the price of the question itself, and the number to use when comparing models.',
 peak:'The highest power reached at any moment during the question.',
 listening:'Recording your voice while the button is held. Almost free.',
 transcribing:'faster-whisper turning that recording into text.',
 answering:'The language model producing an answer and piper speaking it. Usually nearly all of the cost.',
 cpu:'The share of the energy that went through VDD_CORE, the processor rail. The thinking itself.',
 tokens:'How many tokens the model generated. Energy scales almost directly with this.',
 route:'Which Ollama endpoint answered. Models without a chat template fall back from /api/chat to /api/generate.',
 sysprompt:'Whether the bot\\u2019s own system prompt was sent along. It changes the length of the answer, and therefore the energy.',
 tottime:'All the question cycles added up: how long the bot has spent listening, transcribing and answering in total.',
 totcost:'The cost of every question added up, excluding what the bot would have used sitting idle anyway.',
 phases:'How the time split across the three phases, in the same colours as the live graph: green listening, blue transcribing, orange answering. Hover the bar for the exact seconds and energy.'
};
const tip=(t,cls)=>'<i class="i'+(cls?' '+cls:'')+'" tabindex="0" data-tip="'+esc(t)+'"></i>';
const th=(label,key,cls)=>'<th>'+label+(TIP[key]?tip(TIP[key],cls):'')+'</th>';
// Numeric columns: the heading is right-aligned too, so it sits above its own figure.
const thn=(label,key,cls)=>'<th class="n">'+label+(TIP[key]?tip(TIP[key],cls):'')+'</th>';
let lastAsk=null;
function renderAsk(){
  const j=lastAsk;if(!j)return;
  $('out').innerHTML='<table>'+
    '<tr>'+th('model','model')+thn('duration','dur')+thn('peak','peak')+thn('total energy','total')+
    thn('cost of question','cost')+thn('of which cpu','cpu')+
    thn('tokens','tokens')+th('route','route','r')+th('system prompt','sysprompt','r')+'</tr>'+
    '<tr><td>'+esc(j.model)+'</td><td class="n">'+j.seconds.toFixed(2)+' s</td>'+
    '<td class="n">'+j.peak_watts.toFixed(2)+' W</td>'+
    '<td class="n">'+fmtE(j.joules)+'</td><td class="n"><b>'+fmtE(j.joules_above_idle)+'</b></td>'+
    '<td class="n">'+fmtE(j.joules_core)+'</td>'+
    '<td class="n">'+(j.tokens==null?'-':j.tokens)+'</td><td>'+j.endpoint+'</td>'+
    '<td>'+(j.used_system?'yes':'no')+'</td></tr></table>'+
    '<div class="ans">'+esc(j.answer)+'</div>';
}
$('go').onclick=async()=>{
  $('go').disabled=true;$('go').textContent='measuring...';
  try{
    const r=await fetch('api/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:$('m').value,prompt:$('q').value,use_system:$('sys').checked})});
    const j=await r.json();
    lastAsk=j;renderAsk();$('askbox').open=true;
  }catch(e){$('out').textContent='error: '+e;}
  $('go').disabled=false;$('go').textContent='Ask & measure';
};
const clock=t=>new Date(t*1000).toLocaleTimeString();
// Seconds into something readable: 42 s, 3 m 07 s, 1 h 12 m.
function fmtT(x){
  x=Math.round(x||0);
  if(x<60) return x+' s';
  if(x<3600) return Math.floor(x/60)+' m '+String(x%60).padStart(2,'0')+' s';
  return Math.floor(x/3600)+' h '+String(Math.floor(x%3600/60)).padStart(2,'0')+' m';
}
let lastQ=[];
/* Which day the static tab is showing. Everything on that tab reads through
   shownQ(), so the plot, the table and the cloud comparison can never end up
   showing different sets of questions. */
const dayKey=t=>{
  const d=new Date(t*1000);
  return d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+
         '-'+String(d.getDate()).padStart(2,'0');
};
function shownQ(){
  const d=$('qday')?$('qday').value:'';
  return (!d||d==='all')?lastQ:lastQ.filter(q=>dayKey(q.t)===d);
}
function paintQDays(){
  const sel=$('qday'); if(!sel) return;
  const had=sel.value;
  const days=[...new Set(lastQ.map(q=>dayKey(q.t)))].sort().reverse();
  sel.innerHTML='<option value="all">all days</option>'+
    days.map(d=>'<option>'+d+'</option>').join('');
  /* Default to the newest day rather than everything: with months of questions
     "all" is a wall, and the day you just tested is what you came to look at. */
  sel.value=(had&&[...sel.options].some(o=>o.value===had))?had:(days[0]||'all');
  const n=shownQ().length;
  $('qdaynote').textContent=n+' question'+(n===1?'':'s')+
    (sel.value==='all'?' in total':' that day')+
    (days.length>1?' \u00b7 '+days.length+' days logged':'');
}

// Colours are assigned in the order models first appear, so a run with one
// model stays plain and a comparison run separates itself.
const PALETTE=['#d2691e','#3e6ebe','#4aa05a','#9c4bb0','#c2255c','#0f8a8a','#8a6d1f'];
/* Her site works in blue, red and green on cream, so the dots borrow those
   first and only reach for anything else when there are more models than that. */
const PALETTE_PAPER=['#162dab','#d11305','#006951','#000','#828282','#7a2f8f','#b06a00'];
const isPaper=()=>document.documentElement.classList.contains('paper');
/* Every colour the drawn charts use, in one place, so both skins stay honest. */
function SKIN(){
  return isPaper()
    ? {bg:'none', edge:'#000', grid:'rgba(0,0,0,.13)', axis:'#000',
       label:'#000', dim:'#000', radius:'0',
       font:'"Courier New",Courier,monospace'}
    : {bg:'#fffdf5', edge:'#c9b98a', grid:'#e6dcc0', axis:'#c9b98a',
       label:'#6b5b3a', dim:'#9a8a6a', radius:'10px',
       font:'system-ui,sans-serif'};
}
function modelColours(rows){
  const m={},out={};let i=0;
  const pal=isPaper()?PALETTE_PAPER:PALETTE;
  rows.forEach(q=>{const k=q.model||'?';if(!(k in m)){m[k]=pal[i%pal.length];i++;}});
  Object.keys(m).forEach(k=>out[k]=m[k]);
  return out;
}
// Round an axis maximum up to something a person would choose.
function niceMax(v){
  if(!(v>0)) return 1;
  const p=Math.pow(10,Math.floor(Math.log10(v)));
  for(const s of [1,1.5,2,2.5,3,4,5,7.5,10]) if(v<=s*p) return s*p;
  return 10*p;
}
function renderScatter(){
  /* The selected day only. Shadowing the global keeps one code path instead of
     threading a parameter through everything below. */
  const lastQ=shownQ();
  const box=$('scatter');
  if(!lastQ.length){box.innerHTML='<div class="key">Nothing to plot yet.</div>';return;}
  // Oldest first, so the numbers follow the order you asked them.
  const pts=lastQ.slice().sort((a,b)=>a.t-b.t)
    .map((q,i)=>({n:i+1,q,x:U().f(q.above_idle),y:q.total.seconds}));
  const cols=modelColours(pts.map(p=>p.q));
  const W=1100,H=340,L=64,R=20,T=18,B=48;
  const xm=niceMax(Math.max(...pts.map(p=>p.x))),ym=niceMax(Math.max(...pts.map(p=>p.y)));
  const px=v=>L+(v/xm)*(W-L-R), py=v=>H-B-(v/ym)*(H-T-B);
  const S=SKIN();
  let g='';
  for(let i=0;i<=5;i++){
    const xv=xm*i/5,yv=ym*i/5;
    g+='<line x1="'+px(xv)+'" y1="'+T+'" x2="'+px(xv)+'" y2="'+(H-B)+'" stroke="'+S.grid+'"/>'+
       '<text x="'+px(xv)+'" y="'+(H-B+18)+'" text-anchor="middle" font-size="11" '+
       'font-family="'+S.font+'" fill="'+S.dim+'">'+
       (+xv.toFixed(xm<10?2:0))+'</text>'+
       '<line x1="'+L+'" y1="'+py(yv)+'" x2="'+(W-R)+'" y2="'+py(yv)+'" stroke="'+S.grid+'"/>'+
       '<text x="'+(L-8)+'" y="'+(py(yv)+4)+'" text-anchor="end" font-size="11" '+
       'font-family="'+S.font+'" fill="'+S.dim+'">'+
       (+yv.toFixed(ym<10?1:0))+'</text>';
  }
  const dots=pts.map(p=>{
    const c=cols[p.q.model||'?'];
    return '<g class="dot" data-n="'+p.n+'">'+
      '<circle cx="'+px(p.x)+'" cy="'+py(p.y)+'" r="7" fill="'+c+'" fill-opacity=".75" stroke="'+c+'"/>'+
      '<text x="'+px(p.x)+'" y="'+(py(p.y)-12)+'" text-anchor="middle" font-size="10" '+
      'font-family="'+S.font+'" fill="'+S.label+'">'+p.n+'</text>'+
      '<title>'+esc('#'+p.n+'  '+(p.q.text||'(not transcribed)')+'\\n'+(p.q.model||'?')+
        '\\n'+fmtE(p.q.above_idle)+'  ·  '+p.y.toFixed(1)+' s')+'</title></g>';
  }).join('');
  const legend=Object.keys(cols).map(k=>
    '<span class="sw" style="background:'+cols[k]+'"></span>'+esc(k)).join(' &nbsp; ');
  box.innerHTML='<svg viewBox="0 0 '+W+' '+H+'" style="width:100%;height:auto;display:block;'+
    'background:'+S.bg+';border:1px solid '+S.edge+';border-radius:'+S.radius+'">'+g+
    '<line x1="'+L+'" y1="'+T+'" x2="'+L+'" y2="'+(H-B)+'" stroke="'+S.axis+'"/>'+
    '<line x1="'+L+'" y1="'+(H-B)+'" x2="'+(W-R)+'" y2="'+(H-B)+'" stroke="'+S.axis+'"/>'+
    '<text x="'+((L+W-R)/2)+'" y="'+(H-6)+'" text-anchor="middle" font-size="12" '+
    'font-family="'+S.font+'" fill="'+S.label+'">'+
    'cost of question ('+U().lbl+')</text>'+
    '<text transform="translate(16,'+((T+H-B)/2)+') rotate(-90)" text-anchor="middle" '+
    'font-size="12" fill="#6b5b3a">duration (seconds)</text>'+
    dots+'</svg><div class="key">'+legend+'</div>';
}
function renderQ(){
  const lastQ=shownQ();          // the selected day only, as in renderScatter
  if(!lastQ.length){
    $('qlog').innerHTML='<div class="key">Nothing yet. Hold the button on the bot and ask something.</div>';
    return;
  }
  // The three phases as one stacked bar, in the same colours as the live graph.
  const phBar=q=>{
    const p=q.phases,names=['listening','transcribing','answering'];
    const tot=Math.max(0.001,names.reduce((s,n)=>s+(p[n]||{}).seconds,0));
    const tt=names.map(n=>n+' '+(p[n]||{}).seconds.toFixed(1)+' s, '+fmtE((p[n]||{}).joules)).join('\\n');
    return '<div class="pbar" title="'+esc(tt)+'">'+names.map(n=>
      '<span style="width:'+((p[n]||{}).seconds/tot*100).toFixed(1)+'%;background:'+PHC(n)+'"></span>'
    ).join('')+'</div>';
  };
  // Same numbering as the scatter plot: oldest question is 1.
  const nr=new Map();
  lastQ.slice().sort((a,b)=>a.t-b.t).forEach((q,i)=>nr.set(q.t,i+1));
  const by={};
  lastQ.forEach(q=>{const k=q.source||'unknown';(by[k]=by[k]||[]).push(q);});
  const sum=Object.keys(by).sort().map(k=>{
    const a=by[k],n=a.length;
    const tot=a.reduce((s,q)=>s+q.total.seconds,0);
    const totJ=a.reduce((s,q)=>s+q.above_idle,0);
    return '<tr><td>'+k+'</td><td class="n">'+n+'</td><td class="n">'+fmtT(tot)+'</td>'+
           '<td class="n">'+(tot/n).toFixed(1)+' s</td><td class="n">'+fmtE(totJ)+'</td>'+
           '<td class="n"><b>'+fmtE(totJ/n)+'</b></td></tr>';
  }).join('');
  const allT=lastQ.reduce((s,q)=>s+q.total.seconds,0);
  const allJ=lastQ.reduce((s,q)=>s+q.above_idle,0);
  $('qlog').innerHTML=
    '<div class="key" style="font-size:14px;margin-bottom:2px"><b>'+lastQ.length+
    ' question'+(lastQ.length===1?'':'s')+'</b> &middot; <b>'+fmtT(allT)+
    '</b> spent answering in total &middot; <b>'+fmtE(allJ)+'</b> of question cost in total</div>'+
    '<table><tr>'+th('power source','source')+'<th class="n">questions</th>'+
    thn('total time','tottime')+thn('avg duration','dur')+thn('total cost','totcost')+
    thn('avg cost of question','cost','r')+'</tr>'+sum+'</table>'+
    '<table style="margin-top:30px"><tr><th>#</th><th>time</th><th>question</th>'+th('model','model')+th('source','source')+
    thn('battery','batt')+thn('total','total')+thn('cost of question','cost')+
    thn('peak','peak')+th('phases','phases','r')+'</tr>'+
    lastQ.map(q=>'<tr>'+
      '<td>'+(nr.get(q.t)||'')+'</td>'+
      '<td>'+clock(q.t)+'</td>'+
      '<td class="q">'+esc(q.text||'-')+'</td>'+
      '<td>'+esc(q.model)+'</td>'+
      '<td>'+esc(q.source||'-')+'</td>'+
      '<td class="n">'+(q.battery_percent==null?'-':q.battery_percent.toFixed(1)+' %')+'</td>'+
      '<td class="n">'+q.total.seconds.toFixed(1)+' s / '+fmtE(q.total.joules)+'</td>'+
      '<td class="n"><b>'+fmtE(q.above_idle)+'</b></td>'+
      '<td class="n">'+q.total.peak.toFixed(2)+' W</td>'+
      '<td>'+phBar(q)+'</td>'+
    '</tr>').join('')+'</table>'+
    // Same key as under the live graph, so the bars read the same way.
    '<div class="key" style="margin-top:8px">phases:'+
    '<span class="sw" style="background:'+PHC('listening')+';margin-left:10px"></span>listening'+
    '<span class="sw" style="background:'+PHC('transcribing')+';margin-left:12px"></span>transcribing'+
    '<span class="sw" style="background:'+PHC('answering')+';margin-left:12px"></span>answering'+
    '</div>';
}
async function qlog(){
  try{
    const r=await fetch('api/questions');const j=await r.json();
    lastQ=j.questions;paintQDays();renderQ();renderScatter();paintPhaseBar();
    if(!$('tab-report').hidden) loadReport();
  }catch(e){}
}
const qs=()=>'unit='+encodeURIComponent($('unit').value)+
             '&panel='+encodeURIComponent($('panel').value||'5')+
             '&day='+encodeURIComponent(reportDay());
/* An empty day means every day, which keeps the old behaviour as the "all"
   option rather than as a special case threaded through everything. */
const reportDay=()=>{
  const v=$('rday')?$('rday').value:'';
  return (!v||v==='all')?'':v;
};
/* The raw files, kept as a lookup rather than printed as a list.
   A list of every file ever written is a dump, not an offer: it sat on the
   questions tab naming days you were not looking at. The same links are more
   use attached to the thing they belong to, so the day you have open on the
   history tab offers its own csv, and the questions table offers the questions. */
let fileMap={}, fileDir='';
async function files(){
  try{
    const j=await (await fetch('api/files')).json();
    fileMap={}; (j.files||[]).forEach(f=>{ fileMap[f.name]=f.kb; });
    fileDir=j.dir||'';
  }catch(e){}
  paintQFiles(); paintDayFile();
}
function paintQFiles(){
  const el=$('qfiles'); if(!el) return;
  const kb=fileMap['questions.jsonl'];
  el.innerHTML='Download these questions as '+
    '<a href="download/questions.csv" download>csv</a> for a spreadsheet, or as '+
    '<a href="download/questions.jsonl" download>jsonl</a> with every field kept'+
    (kb?' ('+kb+' kB)':'')+'.';
}
function paintDayFile(){
  const el=$('dayfile'); if(!el) return;
  const d=$('day')?$('day').value:'';
  const name='power-'+d+'.csv', kb=fileMap[name];
  if(!d||!kb){ el.textContent=''; return; }
  el.innerHTML='<a href="download/'+encodeURIComponent(name)+'" download>download this day</a> '+
    '<span style="opacity:.7">('+kb+' kB, one row per second)</span>';
}
// The Report tab shows the finished document inline, so you can read it before
// deciding to hand it over.
// The report, rendered straight into the page so it scrolls with everything
// else. The downloadable file is built separately on the server, from the same
// numbers, so the two stay in step.
function paintRDays(){
  const sel=$('rday'); if(!sel) return;
  const had=sel.value;
  const days=[...new Set(lastQ.map(q=>dayKey(q.t)))].sort().reverse();
  sel.innerHTML='<option value="all">all days</option>'+
    days.map(d=>'<option>'+d+'</option>').join('');
  /* The whole set is the honest default here: a report is the thing you hand
     over, and handing over one day without saying so would be a smaller claim
     than it looks. Pick a day deliberately if you want one. */
  sel.value=(had&&[...sel.options].some(o=>o.value===had))?had:'all';
}
function loadReport(){
  paintRDays();
  const day=reportDay();
  $('dl').href='download/report.html?'+qs();
  $('dl').setAttribute('download',
    'solarbot-energy-report'+(day?'-'+day:'')+'.html');
  const box=$('reportbody');
  const shown=day?lastQ.filter(q=>dayKey(q.t)===day):lastQ;
  if(!shown.length){box.innerHTML='<div class="key">'+(day
    ?'Nothing was logged on '+day+'.'
    :'No questions recorded yet. Hold the button on the bot and ask something.')+
    '</div>';return;}
  const rows=shown.slice().sort((a,b)=>b.t-a.t);
  const group=(key)=>{
    const by={};rows.forEach(q=>{const k=q[key]||'unknown';(by[k]=by[k]||[]).push(q);});
    return by;
  };
  const allT=rows.reduce((s,q)=>s+q.total.seconds,0);
  const allJ=rows.reduce((s,q)=>s+q.above_idle,0);

  const bySrc=group('source');
  const srcRows=Object.keys(bySrc).sort().map(k=>{
    const a=bySrc[k],t=a.reduce((s,q)=>s+q.total.seconds,0),j=a.reduce((s,q)=>s+q.above_idle,0);
    return '<tr><td>'+esc(k)+'</td><td class="n">'+a.length+'</td><td class="n">'+fmtT(t)+
           '</td><td class="n">'+(t/a.length).toFixed(1)+' s</td><td class="n">'+fmtE(j)+
           '</td><td class="n"><b>'+fmtE(j/a.length)+'</b></td></tr>';
  }).join('');

  const byMod=group('model');
  const modRows=Object.keys(byMod).sort((a,b)=>byMod[b].length-byMod[a].length).map(k=>{
    const a=byMod[k],j=a.reduce((s,q)=>s+q.above_idle,0);
    return '<tr><td>'+esc(k)+'</td><td class="n">'+a.length+'</td>'+
           '<td class="n"><b>'+fmtE(j/a.length)+'</b></td></tr>';
  }).join('');

  const cards=rows.map(q=>{
    const bars=['listening','transcribing','answering'].map(n=>
      '<span><span class="sw" style="background:'+PHC(n)+'"></span>'+n+': '+
      (q.phases[n]||{}).seconds.toFixed(1)+' s, '+fmtE((q.phases[n]||{}).joules)+'</span>').join('');
    return '<div class="rp"><div><span class="when">'+esc(q.iso||'')+'</span>'+
      '<span class="tag">'+esc(q.model||'?')+'</span><span class="tag">'+esc(q.source||'?')+'</span>'+
      (q.battery_percent==null?'':'<span class="tag">battery '+q.battery_percent.toFixed(1)+' %</span>')+
      '</div><blockquote>'+esc(q.text||'(not transcribed)')+'</blockquote>'+
      '<div class="cost"><b>'+fmtE(q.above_idle)+'</b> cost of this question'+
      '<span class="sep">&middot;</span>'+q.total.seconds.toFixed(1)+' s'+
      '<span class="sep">&middot;</span>peak '+q.total.peak.toFixed(2)+' W</div>'+
      '<div class="bars">'+bars+'</div></div>';
  }).join('');

  box.innerHTML=
    '<div class="key" style="font-size:14px">Generated from '+rows.length+' question'+
    (rows.length===1?'':'s')+' &middot; <b>'+fmtT(allT)+'</b> spent answering in total &middot; <b>'+
    fmtE(allJ)+'</b> of question cost in total &middot; energy shown in '+U().lbl+'</div>'+
    '<h2 style="margin-top:22px">By power source</h2>'+
    '<table><tr><th>source</th><th class="n">questions</th><th class="n">total time</th>'+
    '<th class="n">avg duration</th><th class="n">total cost</th>'+
    '<th class="n">avg cost of a question</th></tr>'+srcRows+'</table>'+
    '<h2 style="margin-top:26px">By model</h2>'+
    '<table><tr><th>model</th><th class="n">questions</th>'+
    '<th class="n">avg cost of a question</th></tr>'+modRows+'</table>'+
    '<h2 style="margin-top:26px">Every question</h2>'+cards+
    '<h2 style="margin-top:26px">What these numbers mean</h2>'+
    '<div class="key"><b>Cost of this question</b> is the energy the question '+
    'added on top of simply having the bot switched on. The Pi draws about 2 W doing nothing, so a '+
    'question that takes ten seconds spends roughly 20 J just existing. That part is subtracted, '+
    'which leaves what the thinking actually cost.<br><br>'+
    'Measured from the Raspberry Pi 5&rsquo;s own PMIC, which reports the board&rsquo;s internal '+
    'rails: processor, memory, wifi. It does not include the Whisplay HAT&rsquo;s screen and speaker, '+
    'anything on USB, or the PiSugar&rsquo;s conversion losses, so the real drain on the battery is '+
    'somewhat higher.</div>';
}
function showTab(name){
  document.querySelectorAll('.tabs button').forEach(b=>b.classList.toggle('on',b.dataset.tab===name));
  document.querySelectorAll('.panel').forEach(p=>p.hidden=(p.id!=='tab-'+name));
  try{localStorage.setItem('solarbot-tab',name);}catch(e){}
  if(name==='report') loadReport();
  if(name==='static' && $('cmpbox').open) loadCompare();
  if(name==='history'){ loadDays(); }
}
/* ---- history ----------------------------------------------------------
   The live graph only ever holds the last few minutes. Everything older is
   on disk and was, until now, only downloadable. This draws a whole day. */
let hist=null;
async function loadDays(){
  try{
    const j=await (await fetch('api/days')).json();
    const sel=$('day'), had=sel.value;
    sel.innerHTML=(j.days||[]).map(d=>'<option>'+d+'</option>').join('');
    if(had&&(j.days||[]).includes(had)) sel.value=had;
    if(sel.value) loadDay();
  }catch(e){}
}
async function loadDay(){
  const d=$('day').value; if(!d) return;
  try{
    hist=await (await fetch('api/history?day='+encodeURIComponent(d))).json();
  }catch(e){ hist=null; }
  paintDayFile();
  showDay();
}
function showDay(){
  const h=hist;
  if(!h||h.error||!h.points||!h.points.length){
    $('daynote').textContent=h&&h.error?h.error:'nothing logged that day';
    $('hcov').textContent=$('hwh').textContent=$('hmean').textContent=
      $('hpeak').textContent=$('hbat').textContent='-';
    drawHist(); return;
  }
  const hrs=h.seconds/3600;
  $('hcov').textContent=hrs>=1?hrs.toFixed(1)+' h':Math.round(h.seconds/60)+' min';
  $('hwh').textContent=h.wh.toFixed(2);
  $('hmean').textContent=h.mean.toFixed(2);
  $('hpeak').textContent=h.peak.toFixed(2);
  const b=h.battery||{};
  $('hbat').textContent=(b.first==null||b.last==null)?'-'
    :(b.first.toFixed(0)+'% \u2192 '+b.last.toFixed(0)+'%');
  $('daynote').textContent=(h.seconds/864).toFixed(0)+'% of the day covered';
  drawHist();
}
function drawHist(){
  const c=$('hc'); if(!c) return;
  const x=c.getContext('2d'), W=c.width, H=c.height, L=54, R=14, T=12, B=30;
  const S=SKIN();
  x.clearRect(0,0,W,H);
  if(S.bg!=='none'){x.fillStyle=S.bg;x.fillRect(0,0,W,H);}
  const h=hist;
  if(!h||!h.points||!h.points.length){
    x.fillStyle=S.dim;x.font='13px '+S.font;
    x.fillText('nothing logged that day',L,H/2); return;
  }
  /* Always the whole day, so two days can be compared by eye and a short
     stretch of logging looks as short as it was. */
  const t0=h.points[0][0]-(h.points[0][0]%86400)+(new Date(h.day+'T00:00:00')).getTimezoneOffset()*60;
  const day0=Math.min(h.points[0][0], t0), day1=day0+86400;
  const peak=Math.max(1,Math.max.apply(null,h.points.map(p=>p[3])))*1.1;
  const px=t=>L+(t-day0)/86400*(W-L-R), py=w=>H-B-(w/peak)*(H-T-B);
  x.strokeStyle=S.grid;x.lineWidth=1;x.fillStyle=S.dim;x.font='11px '+S.font;
  for(let hh=0;hh<=24;hh+=3){
    const xx=px(day0+hh*3600);
    x.beginPath();x.moveTo(xx,T);x.lineTo(xx,H-B);x.stroke();
    x.textAlign='center';x.fillText((hh<10?'0':'')+hh+':00',xx,H-B+16);
  }
  x.textAlign='right';
  for(let i=0;i<=4;i++){
    const w=peak*i/4,yy=py(w);
    x.beginPath();x.moveTo(L,yy);x.lineTo(W-R,yy);x.stroke();
    x.fillText(w.toFixed(1),L-6,yy+4);
  }
  /* The per-minute peak sits behind the mean, so a burst that lasted seconds
     is still visible instead of being averaged into nothing. */
  const gap=(a,b)=>b-a>120;   // more than two minutes apart is a gap
  const line=(idx,style,width)=>{
    x.strokeStyle=style;x.lineWidth=width;x.beginPath();
    let pen=false;
    for(let i=0;i<h.points.length;i++){
      const p=h.points[i];
      if(pen&&gap(h.points[i-1][0],p[0])){x.stroke();x.beginPath();pen=false;}
      const X=px(p[0]),Y=py(p[idx]);
      if(pen) x.lineTo(X,Y); else {x.moveTo(X,Y);pen=true;}
    }
    x.stroke();
  };
  const paper=isPaper();
  const cPeak=paper?'rgba(0,0,0,.30)':'rgba(210,105,30,.40)';
  const cTot =paper?'#000':'#6b5b3a';
  const cCpu =paper?'#162dab':'#3e6ebe';
  line(3, cPeak, 1);              // the highest single second in each minute
  line(1, cTot , 1.6);            // the average of that minute, everything
  line(2, cCpu , 1.2);            // the average of that minute, cpu only
  x.strokeStyle=S.axis;x.lineWidth=1;
  x.beginPath();x.moveTo(L,T);x.lineTo(L,H-B);x.lineTo(W-R,H-B);x.stroke();
  /* Three lines is two more than anyone guesses, so they name themselves. */
  x.textAlign='left';x.font='11px '+S.font;
  x.fillStyle=cPeak;x.fillText('peak W',L+6,T+12);
  x.fillStyle=cTot ;x.fillText('total W',L+62,T+12);
  x.fillStyle=cCpu ;x.fillText('cpu W',L+122,T+12);
}
$('day').onchange=loadDay;

/* ---- the cloud comparison ---------------------------------------------
   The page measures what a question cost on this Pi. ecocost estimates what
   the same question would have cost in a data centre. Putting the two next to
   each other is the only honest use of it: it cannot measure, and it has never
   heard of the model running here. */
let cmp=null;
async function loadCompare(){
  try{
    const m=$('cmodel').value;
    cmp=await (await fetch('api/compare'+(m?'?model='+encodeURIComponent(m):''))).json();
  }catch(e){ cmp=null; }
  renderCompare();
}
function renderCompare(){
  const box=$('compare'), note=$('comparenote');
  if(!cmp||!cmp.available){
    note.textContent='Not available: the ecocost package is not installed on the Pi.';
    box.innerHTML=''; $('cmodel').innerHTML=''; return;
  }
  const sel=$('cmodel');
  if(!sel.options.length){
    sel.innerHTML=(cmp.models||[]).map(m=>'<option'+(m===cmp.model?' selected':'')+'>'+m+'</option>').join('');
  }
  note.innerHTML='Your questions priced as if a data centre had answered them, by '+
    '<a href="https://gooey.ai/ecocost" target="_blank" rel="noopener">ecocost</a>. '+
    'It works from token counts alone, so the answer length is derived from how long '+
    'the answering phase took at <b>'+cmp.tokens_per_second+' tokens a second</b>, measured '+
    'on this Pi for '+esc(cmp.bot_model||'the local model')+'. '+
    'The range is the estimate\u2019s own: ecocost does not know which chip, which data '+
    'centre or which grid answered, and says so.';
  const dag=$('qday')?$('qday').value:'all';
  const rows=(cmp.questions||[]).filter(r=>dag==='all'||dayKey(r.t)===dag);
  if(!rows.length){ box.innerHTML='<div class="key">Nothing to compare on that day.</div>'; return; }
  const here=rows.reduce((s,r)=>s+(r.measured||0),0);
  const there=rows.reduce((s,r)=>s+(r.joules||0),0);
  const lo=rows.reduce((s,r)=>s+(r.min||0),0), hi=rows.reduce((s,r)=>s+(r.max||0),0);
  const ratio=here>0?there/here:0;
  box.innerHTML=
    '<div class="key" style="font-size:14px;margin:12px 0 2px">'+
    'These <b>'+rows.length+'</b> questions cost <b>'+fmtE(here)+'</b> here. On <b>'+
    esc(cmp.model)+'</b> the same questions come out at <b>'+fmtE(there)+'</b>'+
    (ratio?', about <b>'+ratio.toFixed(0)+' times as much</b>':'')+
    ' \u2014 somewhere between '+fmtE(lo)+' and '+fmtE(hi)+'.</div>'+
    '<table style="margin-top:14px"><tr><th>time</th><th>question</th>'+
    '<th class="n">here</th><th class="n">'+esc(cmp.model)+'</th>'+
    '<th class="n">range</th><th class="n">times more</th>'+
    '<th class="n">CO\u2082</th><th class="n">water</th></tr>'+
    rows.slice().sort((a,b)=>b.t-a.t).map(r=>{
      const x=r.measured>0?(r.joules/r.measured):0;
      return '<tr>'+
      '<td>'+clock(r.t)+'</td>'+
      '<td class="q">'+esc(r.text||'-')+'</td>'+
      '<td class="n">'+fmtE(r.measured||0)+'</td>'+
      '<td class="n"><b>'+fmtE(r.joules)+'</b></td>'+
      '<td class="n" title="'+esc(r.tokens_in+' tokens in, '+r.tokens_out+
        ' out \u00b7 confidence: '+r.confidence)+'">'+fmtE(r.min)+' \u2013 '+fmtE(r.max)+'</td>'+
      '<td class="n">'+(x?x.toFixed(0)+'\u00d7':'-')+'</td>'+
      '<td class="n">'+r.co2_g.toFixed(3)+' g</td>'+
      '<td class="n">'+r.water_ml.toFixed(2)+' mL</td>'+
      '</tr>';
    }).join('')+'</table>';
}
$('cmodel').onchange=loadCompare;
$('qday').onchange=()=>{ paintQDays();renderQ();renderScatter();renderCompare(); };
$('rday').onchange=loadReport;
$('cmpbox').ontoggle=()=>{ if($('cmpbox').open) loadCompare(); };

const redraw=()=>{
  unitNote();renderQ();renderScatter();renderAsk();poll();
  if(!$('tab-report').hidden) loadReport();
};
$('unit').onchange=redraw;$('panel').oninput=redraw;
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>showTab(b.dataset.tab));
let startTab='live';
try{startTab=localStorage.getItem('solarbot-tab')||'live';}catch(e){}
if(!document.getElementById('tab-'+startTab)) startTab='live';
showTab(startTab);
models();poll();qlog();files();unitNote();
$('face').onclick=stepFace;
$('face').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();stepFace();}};
paintFace();
setInterval(()=>{faceStep++;paintFace();},420);
/* ---- the skin switch -------------------------------------------------
   Paper is on unless you turn it off, so the new look is what you see and one
   click puts the old one back beside it for comparison. */
function paintSkin(){
  const on=localStorage.getItem('skin')!=='plain';
  document.documentElement.classList.toggle('paper',on);
  $('skin').textContent=on?'skin: paper':'skin: original';
  paintPhaseBar();
}
/* The bar Ambika liked in the report, put under the live graph as well.
   Once the bot has answered something it shows how that question split across
   the phases; before that it stands in as the key for the bands on the graph,
   drawn in equal parts and faded so it does not claim to be a measurement. */
function paintPhaseBar(){
  const box=$('phasebar'); if(!box) return;
  const names=['listening','transcribing','answering'];
  let q=null;
  for(const r of (lastQ||[])) if(!q||r.t>q.t) q=r;
  const secs=names.map(n=>q?(((q.phases||{})[n]||{}).seconds||0):0);
  const tot=secs.reduce((a,b)=>a+b,0);
  const live=tot>0.05;
  const w=names.map((n,i)=>live?secs[i]/tot*100:100/names.length);
  const tip=names.map((n,i)=>n+' '+secs[i].toFixed(1)+' s, '+
    fmtE((((q||{}).phases||{})[n]||{}).joules||0)).join(' \u00b7 ');
  box.innerHTML=
    '<div class="pbar"'+(live?' title="'+esc(tip)+'"':' style="opacity:.42"')+'>'+
    names.map((n,i)=>'<span style="width:'+w[i].toFixed(1)+'%;background:'+
      PHC(n)+'"></span>').join('')+'</div>'+
    '<div class="key">phases:'+
    names.map((n,i)=>'<span class="sw" style="background:'+PHC(n)+
      ';margin-left:'+(i?12:10)+'px"></span>'+n).join('')+
    (live?'':'<span style="opacity:.6;margin-left:16px">'+
      '\u2014 nothing asked yet, so these are only the colours</span>')+
    '</div>';
}
$('skin').onclick=()=>{
  localStorage.setItem('skin',localStorage.getItem('skin')==='plain'?'paper':'plain');
  paintSkin();draw();drawHist();
};
paintSkin();
/* what this page cost to load, stated the way her site states it */
addEventListener('load',()=>{
  try{
    let b=(performance.getEntriesByType('navigation')[0]||{}).transferSize||0;
    for(const r of performance.getEntriesByType('resource')) b+=r.transferSize||0;
    if(b) $('pagecost').textContent='this page: '+(b/1024).toFixed(1)+' kB';
  }catch(e){}
});
setInterval(poll,1000);setInterval(qlog,2000);setInterval(files,30000);
</script>
"""


def build_report(rows, unit="J", panel="5", toolbar=False, day=""):
    """A standalone page you can open, read, hand over, or print to PDF.

    The spreadsheet export is for analysis. This is for showing someone.
    With toolbar=True it is served for viewing at /report and gets navigation
    across the top; without, it is the file you download, which has to work
    on its own once it leaves the Pi, so the links are left out.
    """
    try:
        panel_w = max(0.1, float(panel))
    except ValueError:
        panel_w = 5.0
    conv = {
        "J":   ("J",   1, lambda j: j),
        "mWh": ("mWh", 2, lambda j: j / 3.6),
        "mAh": ("mAh", 3, lambda j: j / BATTERY_NOMINAL_V / 3.6),
        "pct": ("% of a charge", 3, lambda j: j / BATTERY_JOULES * 100),
        "sun": ("s of sun", 1, lambda j: j / panel_w),
    }.get(unit, ("J", 1, lambda j: j))
    lbl, dec, fn = conv

    def e(j):
        return "-" if j is None else ("%.*f %s" % (dec, fn(j), lbl))

    esc = html.escape
    by = {}
    for r in rows:
        by.setdefault(r.get("source") or "unknown", []).append(r)

    def fmt_t(x):
        x = int(round(x or 0))
        if x < 60:
            return "%d s" % x
        if x < 3600:
            return "%d m %02d s" % (x // 60, x % 60)
        return "%d h %02d m" % (x // 3600, x % 3600 // 60)

    summary = ""
    for src in sorted(by):
        a = by[src]
        tot = sum(x.get("total", {}).get("seconds", 0) for x in a)
        totj = sum(x.get("above_idle", 0) for x in a)
        summary += ("<tr><td>%s</td><td>%d</td><td>%s</td><td>%.1f s</td>"
                    "<td>%s</td><td><b>%s</b></td></tr>"
                    % (esc(src), len(a), fmt_t(tot), tot / len(a),
                       e(totj), e(totj / len(a))))

    all_t = sum(x.get("total", {}).get("seconds", 0) for x in rows)
    all_j = sum(x.get("above_idle", 0) for x in rows)

    bar = ""
    if toolbar:
        qs = "unit=%s&panel=%s&day=%s" % (urllib.parse.quote(unit),
                                          urllib.parse.quote(str(panel)),
                                          urllib.parse.quote(day))
        units = " ".join(
            '<a href="/report?unit=%s&panel=%s&day=%s"%s>%s</a>'
            % (u, urllib.parse.quote(str(panel)), urllib.parse.quote(day),
               ' class="on"' if u == unit else "", name)
            for u, name in (("J", "joules"), ("mWh", "mWh"), ("mAh", "mAh"),
                            ("pct", "% of charge"), ("sun", "sun seconds")))
        bar = ("""<nav>
  <a href="/">&larr; back to the live meter</a>
  <span class="units">show as: %s</span>
  <span class="right">
    <a href="/download/report.html?%s" download="solarbot-energy-report.html">download this report</a>
    <button onclick="print()">print or save as PDF</button>
  </span>
</nav>""" % (units, qs))

    models = {}
    for r in rows:
        models.setdefault(r.get("model") or "?", []).append(r.get("above_idle", 0))
    permodel = ""
    for m in sorted(models, key=lambda k: -len(models[k])):
        v = models[m]
        avg = sum(v) / len(v)
        permodel += ("<tr><td>%s</td><td>%d</td><td><b>%s</b></td></tr>"
                     % (esc(m), len(v), e(avg)))

    items = ""
    for r in reversed(rows):
        p, tot = r.get("phases", {}), r.get("total", {})
        bat = r.get("battery_percent")
        items += """
<article>
  <header><span class="when">%s</span>
    <span class="tag">%s</span><span class="tag">%s</span>%s</header>
  <blockquote>%s</blockquote>
  <div class="cost"><b>%s</b> <span>cost of this question</span>
    <span class="sep">&middot;</span> %.1f s <span class="sep">&middot;</span> peak %.2f W</div>
  <div class="bars">%s</div>
</article>""" % (
            esc(r.get("iso") or ""),
            esc(r.get("model") or "?"),
            esc(r.get("source") or "?"),
            "" if bat is None else '<span class="tag">battery %.1f %%</span>' % bat,
            esc(r.get("text") or "(not transcribed)"),
            e(r.get("above_idle")),
            tot.get("seconds", 0), tot.get("peak", 0),
            "".join(
                '<div class="bar"><span class="%s"></span><label>%s: %.1f s, %s</label></div>'
                % (name, name, (p.get(name) or {}).get("seconds", 0),
                   e((p.get(name) or {}).get("joules")))
                for name in ("listening", "transcribing", "answering")),
        )

    return """<!doctype html>
<meta charset="utf-8"><title>Solarbot energy report</title>
<style>
 body{font:15px/1.6 system-ui,sans-serif;color:#2b2320;background:#fff5d1;margin:0;padding:32px}
 main{max-width:860px;margin:0 auto}
 h1{font-size:26px;margin:0 0 4px} h2{font-size:17px;margin:30px 0 10px}
 .sub{opacity:.7;font-size:13px;margin-bottom:24px}
 table{border-collapse:collapse;width:100%%;font-size:14px;background:#fffdf5;
       border:1px solid #c9b98a;border-radius:8px;overflow:hidden}
 th,td{text-align:left;padding:7px 10px;border-bottom:1px solid #e6dcc0}
 th{font-size:11px;text-transform:uppercase;letter-spacing:.06em;opacity:.65}
 tr:last-child td{border-bottom:0}
 article{background:#fffdf5;border:1px solid #c9b98a;border-radius:10px;
         padding:14px 16px;margin:10px 0;break-inside:avoid}
 header{font-size:12px;opacity:.75;margin-bottom:8px}
 .when{margin-right:8px}
 .tag{display:inline-block;background:#f0e6c8;border-radius:20px;padding:1px 9px;margin-right:5px}
 blockquote{margin:0 0 10px;font-size:17px;font-style:italic}
 .cost{font-size:13px} .cost b{font-size:20px;font-style:normal}
 .cost span{opacity:.7} .sep{margin:0 6px;opacity:.4}
 .bars{margin-top:10px;display:flex;gap:14px;flex-wrap:wrap;font-size:12px}
 .bar{display:flex;align-items:center;gap:6px}
 .bar span{width:11px;height:11px;border-radius:3px;display:inline-block}
 .listening{background:#4aa05a}.transcribing{background:#3e6ebe}.answering{background:#d2691e}
 .note{font-size:13px;opacity:.85}
 nav{background:#fffdf5;border:1px solid #c9b98a;border-radius:10px;padding:9px 14px;
     margin-bottom:22px;font-size:13px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 nav a{color:inherit} nav .units a{margin-left:7px;opacity:.6;text-decoration:none}
 nav .units a.on{opacity:1;font-weight:600;text-decoration:underline}
 nav .right{margin-left:auto;display:flex;gap:12px;align-items:center}
 nav button{font:inherit;border:1px solid #d2691e;background:#d2691e;color:#fff;
            border-radius:7px;padding:5px 12px;cursor:pointer}
 @media print{body{background:#fff;padding:0}article,table{border-color:#bbb}nav{display:none}}
</style>
<main>
%s
<h1>Solarbot energy report</h1>
<div class="sub">%s &middot; generated %s &middot; %d question%s &middot; energy shown in %s</div>

<h2>Totals</h2>
<p class="note"><b>%s</b> spent answering across all questions, costing <b>%s</b> in total
on top of what the bot would have used sitting idle anyway.</p>

<h2>By power source</h2>
<table><tr><th>source</th><th>questions</th><th>total time</th><th>avg duration</th>
<th>total cost</th><th>avg cost of a question</th></tr>%s</table>

<h2>By model</h2>
<table><tr><th>model</th><th>questions</th><th>avg cost of a question</th></tr>%s</table>

<h2>Every question</h2>
%s

<h2>What these numbers mean</h2>
<p class="note"><b>Cost of this question</b> is the energy the question added on top of
simply having the bot switched on. The Pi draws about 2 W doing nothing, so a question
that takes ten seconds spends roughly 20 J just existing. That part is subtracted, which
leaves what the thinking actually cost.</p>
<p class="note">Measured from the Raspberry Pi 5's own PMIC, which reports the board's
internal rails: processor, memory, wifi. It does not include the Whisplay HAT's screen and
speaker, anything on USB, or the PiSugar's conversion losses, so the real drain on the
battery is somewhat higher.</p>
</main>""" % (bar, ("All days" if not day else day),
              time.strftime("%Y-%m-%d %H:%M"), len(rows),
              "" if len(rows) == 1 else "s", esc(lbl),
              fmt_t(all_t), e(all_j),
              summary or "<tr><td colspan=6>nothing logged yet</td></tr>",
              permodel or "<tr><td colspan=3>nothing logged yet</td></tr>",
              items or "<p class='note'>No questions recorded yet.</p>")


# The five state drawings, the same ones the bot shows on its own screen.
# They live in faces/<state>/<0-2>.png next to this script. Serving them is the
# only file serving this program does, so the path is checked against a fixed
# list rather than being resolved against the filesystem.
FACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces")
FACE_STATES = ("idle", "listening", "detecting", "recognizing", "answering")


def read_day(day):
    """One day of the log, averaged to the minute.

    A day is 86400 rows, which is four megabytes down the wire and far more
    detail than anyone reads on a chart a thousand pixels wide. Averaging to
    the minute gives 1440 points, and the peak within each minute is kept
    alongside the mean so a short burst is still visible rather than smoothed
    away.

    Minutes with no reading are left out entirely instead of being filled in.
    That matters here: a gap means the bot was off or the logger was not
    running, and that is exactly the thing worth seeing.
    """
    if not day or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        return {"day": day, "error": "no such day"}
    path = os.path.join(DATA_DIR, "power-%s.csv" % day)
    if not os.path.exists(path):
        return {"day": day, "error": "no such day"}

    buckets = {}
    joules = 0.0
    seconds = 0
    peak = 0.0
    bat_first = bat_last = None
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.rstrip("\n").split(",")
                if len(parts) < 4 or parts[0] == "time":
                    continue
                try:
                    t = int(float(parts[1]))
                    total = float(parts[2])
                    cpu = float(parts[3])
                except ValueError:
                    continue
                # One row is one second, so watts and joules are the same number.
                joules += total
                seconds += 1
                if total > peak:
                    peak = total
                try:
                    lvl = float(parts[4])
                    if bat_first is None:
                        bat_first = lvl
                    bat_last = lvl
                except (IndexError, ValueError):
                    pass
                m = t - (t % 60)
                b = buckets.get(m)
                if b is None:
                    buckets[m] = [total, cpu, total, 1]   # sum, cpusum, max, n
                else:
                    b[0] += total
                    b[1] += cpu
                    if total > b[2]:
                        b[2] = total
                    b[3] += 1
    except OSError as e:
        return {"day": day, "error": str(e)}

    points = [[m, round(v[0] / v[3], 3), round(v[1] / v[3], 3), round(v[2], 3)]
              for m, v in sorted(buckets.items())]
    return {
        "day": day,
        "points": points,                      # minute, mean W, mean cpu W, peak W
        "seconds": seconds,                    # how much of the day was logged
        "joules": round(joules, 1),
        "wh": round(joules / 3600.0, 3),
        "mean": round(joules / seconds, 3) if seconds else 0,
        "peak": round(peak, 3),
        "battery": {"first": bat_first, "last": bat_last},
    }


def compare_models():
    """The cloud models worth offering in the dropdown.

    The knowledge base holds seventy-odd models, most of them provider variants
    nobody would recognise. These are the families you would actually name when
    asked "what if I had asked ChatGPT instead".
    """
    if ecocost is None:
        return []
    try:
        ids = sorted(ecocost.get_kb().models)
    except Exception:
        return []
    keep = ("gpt-", "claude-", "gemini-", "deepseek-", "grok")
    return [m for m in ids if m.startswith(keep)]


def compare_question(q, model):
    """What one logged question would have cost in a data centre, in joules.

    Returns None when there is honestly nothing to say: no ecocost installed, a
    model it does not know, or a question that never reached the answering
    phase and so has no output to price.
    """
    if ecocost is None:
        return None
    answering = ((q.get("phases") or {}).get("answering") or {}).get("seconds") or 0
    if answering <= 0:
        return None
    text = q.get("text") or ""
    tokens_in = int((len(BOT_SYSTEM) + len(text)) / CHARS_PER_TOKEN) + 1
    tokens_out = max(1, int(round(answering * TOKENS_PER_SECOND)))
    try:
        r = ecocost.estimate(model, input_tokens=tokens_in, output_tokens=tokens_out)
    except Exception:
        return None
    e = r["energy"]
    return {
        # ecocost answers in watt-hours; the rest of this page speaks joules.
        "joules": round(e["value"] * 3600, 1),
        "min": round(e["min"] * 3600, 1),
        "max": round(e["max"] * 3600, 1),
        "co2_g": r["carbon"]["value"],
        "water_ml": r["water"]["value"],
        "confidence": r["confidence"]["level"],
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        """Send one response, and say out loud that it must not be kept.

        Nothing here said anything about caching, and a reply with no
        Cache-Control, no ETag and no Last-Modified lets the browser invent an
        expiry of its own. It does. That is how a page built around a meter
        that moves every second ends up showing yesterday, and how a freshly
        deployed version keeps serving the old one: the request never reaches
        the Pi, so nothing on this side can tell that it happened.

        Everything sent from here is either a live reading or a page built
        around one, so all of it is no-store. The drawn faces are the only
        thing worth keeping and they are written out separately, with a day of
        cache, because the animation swaps between them twice a second.
        """
        body = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if path.startswith("/faces/"):
            parts = path[len("/faces/"):].split("/")
            if (len(parts) != 2 or parts[0] not in FACE_STATES
                    or parts[1] not in ("0.png", "1.png", "2.png")):
                return self._send(404, b"no", "text/plain")
            try:
                with open(os.path.join(FACE_DIR, parts[0], parts[1]), "rb") as fh:
                    body = fh.read()
            except OSError:
                return self._send(404, b"no", "text/plain")
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            # they never change, so let the browser keep them for a day
            self.send_header("Cache-Control", "max-age=86400")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        if path == "/api/samples":
            with samples_lock:
                rows = [[round(t, 3), round(w, 4), round(c, 4), round(p, 4)]
                        for t, w, c, p in samples]
            t0 = rows[0][0] - 5 if rows else 0
            with states_lock:
                st = [[round(t, 3), n] for t, n in states if t >= t0]
            with battery_lock:
                bat = dict(battery)
            with panel_lock:
                pan = dict(panel)
            pan["joules_in"] = round(joules_in, 1)
            return self._send(200, json.dumps({"samples": rows,
                                               "states": st,
                                               "idle": round(idle_watts, 4),
                                               "battery": bat,
                                               "panel": pan,
                                               "session": session_report()}),
                              "application/json")
        if path == "/api/compare":
            q = {}
            if "?" in self.path:
                for part in self.path.split("?", 1)[1].split("&"):
                    k, _, v = part.partition("=")
                    q[k] = urllib.parse.unquote_plus(v)
            model = q.get("model") or COMPARE_MODEL
            out = []
            with questions_lock:
                rows = list(questions)
            for item in rows:
                c = compare_question(item, model)
                if not c:
                    continue
                c.update(t=item.get("t"), iso=item.get("iso"),
                         text=item.get("text"),
                         measured=item.get("above_idle"),
                         seconds=(item.get("total") or {}).get("seconds"))
                out.append(c)
            return self._send(200, json.dumps({
                "available": ecocost is not None,
                "model": model,
                "models": compare_models(),
                "tokens_per_second": TOKENS_PER_SECOND,
                "bot_model": BOT_MODEL,
                "questions": out,
            }), "application/json")
        if path == "/api/days":
            # Which days there is anything to look at, newest first.
            try:
                names = sorted(os.listdir(DATA_DIR), reverse=True)
            except OSError:
                names = []
            days = [n[6:-4] for n in names
                    if n.startswith("power-") and n.endswith(".csv")]
            return self._send(200, json.dumps({"days": days}),
                              "application/json")
        if path == "/api/history":
            q = {}
            if "?" in self.path:
                for part in self.path.split("?", 1)[1].split("&"):
                    k, _, v = part.partition("=")
                    q[k] = urllib.parse.unquote_plus(v)
            return self._send(200, json.dumps(read_day(q.get("day", ""))),
                              "application/json")
        if path in ("/report", "/download/report.html"):
            q = {}
            if "?" in self.path:
                for part in self.path.split("?", 1)[1].split("&"):
                    k, _, v = part.partition("=")
                    q[k] = urllib.parse.unquote_plus(v)
            with questions_lock:
                rows = list(questions)
            # An empty day means every day, which is what the report always was.
            day = q.get("day", "")
            if day:
                rows = [r for r in rows
                        if time.strftime("%Y-%m-%d",
                                         time.localtime(r.get("t", 0))) == day]
            return self._send(200, build_report(rows, q.get("unit", "J"),
                                                q.get("panel", "5"),
                                                toolbar=(path == "/report"),
                                                day=day),
                              "text/html; charset=utf-8")
        if path == "/download/questions.csv":
            with questions_lock:
                rows = list(questions)
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["time", "question", "model", "power_source",
                        "battery_percent", "battery_volts",
                        "total_seconds", "total_joules", "question_cost_joules",
                        "peak_watts", "idle_watts",
                        "listening_seconds", "listening_joules",
                        "transcribing_seconds", "transcribing_joules",
                        "answering_seconds", "answering_joules",
                        "cpu_joules"])
            for q in rows:
                p, tot = q.get("phases", {}), q.get("total", {})

                def g(d, k):
                    return (d or {}).get(k, "")

                w.writerow([
                    q.get("iso") or time.strftime("%Y-%m-%d %H:%M:%S",
                                                  time.localtime(q.get("t", 0))),
                    q.get("text", ""), q.get("model", ""),
                    q.get("source", ""), q.get("battery_percent", ""),
                    q.get("battery_volts", ""),
                    g(tot, "seconds"), g(tot, "joules"), q.get("above_idle", ""),
                    g(tot, "peak"), q.get("idle_watts", ""),
                    g(p.get("listening"), "seconds"), g(p.get("listening"), "joules"),
                    g(p.get("transcribing"), "seconds"), g(p.get("transcribing"), "joules"),
                    g(p.get("answering"), "seconds"), g(p.get("answering"), "joules"),
                    g(tot, "joules_core"),
                ])
            return self._send(200, buf.getvalue(), "text/csv; charset=utf-8")
        if path.startswith("/download/"):
            name = os.path.basename(path[len("/download/"):])
            if name != "questions.jsonl" and not re.fullmatch(
                    r"power-\d{4}-\d{2}-\d{2}\.csv", name):
                return self._send(404, "not found", "text/plain")
            try:
                with open(os.path.join(DATA_DIR, name), "rb") as f:
                    return self._send(200, f.read(), "text/plain; charset=utf-8")
            except Exception:
                return self._send(404, "no data yet", "text/plain")
        if path == "/api/files":
            try:
                names = sorted(os.listdir(DATA_DIR))
            except Exception:
                names = []
            out = []
            for n in names:
                try:
                    out.append({"name": n,
                                "kb": round(os.path.getsize(os.path.join(DATA_DIR, n)) / 1024, 1)})
                except Exception:
                    pass
            return self._send(200, json.dumps({"dir": DATA_DIR, "files": out}),
                              "application/json")
        if path == "/api/questions":
            with questions_lock:
                rows = list(questions)[-40:]
            return self._send(200, json.dumps({"questions": rows[::-1]}),
                              "application/json")
        if path == "/api/models":
            return self._send(200, json.dumps({"models": ollama_models(),
                                               "bot_model": BOT_MODEL,
                                               "system_chars": len(BOT_SYSTEM),
                                               "battery_joules": round(BATTERY_JOULES, 1),
                                               "battery_mah": BATTERY_MAH,
                                               "battery_volts": BATTERY_NOMINAL_V}),
                              "application/json")
        self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path.split("?")[0].rstrip("/") == "/api/session/reset":
            with battery_lock:
                lvl = battery["level"]
            with session_lock:
                session.update({"t": time.time(), "joules": joules_total,
                                "level": lvl})
            return self._send(200, json.dumps(session_report()), "application/json")
        if self.path.split("?")[0].rstrip("/") != "/api/ask":
            return self._send(404, "not found", "text/plain")
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        model = req.get("model") or BOT_MODEL
        prompt = req.get("prompt") or ""
        use_system = bool(req.get("use_system")) and bool(BOT_SYSTEM)
        system = BOT_SYSTEM if use_system else ""

        base = idle_watts
        t0 = time.time()
        answer, endpoint, tokens = ollama_ask(model, prompt, system)
        time.sleep(1.5 / SAMPLE_HZ)          # let the last sample land
        t1 = time.time()

        rows = window(t0, t1)
        j_tot, j_core = energy(rows)
        seconds = t1 - t0
        self._send(200, json.dumps({
            "seconds": seconds,
            "peak_watts": max((r[1] for r in rows), default=0.0),
            "joules": j_tot,
            "joules_above_idle": max(0.0, j_tot - base * seconds),
            "joules_core": j_core,
            "idle_watts": base,
            "tokens": tokens,
            "endpoint": endpoint,
            "model": model,
            "used_system": use_system,
            "answer": answer,
        }), "application/json")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    load_questions()
    print("Loaded %d earlier questions from %s" % (len(questions), QUESTIONS_FILE))
    threading.Thread(target=sampler, daemon=True).start()
    threading.Thread(target=log_tailer, daemon=True).start()
    threading.Thread(target=battery_poller, daemon=True).start()
    time.sleep(1.5)
    print("Power meter on http://0.0.0.0:%d/" % PORT)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
