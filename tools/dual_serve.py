#!/usr/bin/env python3
"""Two Strata servers, one per GPU, on one shared expert arena (Linux, HIP).

    python tools/dual_serve.py [--config strata-q2_0.json] [--arena /mnt/strata-shm/arena.bin]

Instance A (GPU --gpu-a, port --port-a) loads the arena into the MAP_SHARED file and pins it for its GPU.  Instance B
(GPU --gpu-b, port --port-b) maps the same file and only locks it (STRATA_ARENA_PIN_GIB=none): a second process that
pins the same arena is refused by the driver once the first holds most of the RAM.  The file must be on a tmpfs big
enough for the arena (about 32 GiB for Q2_0); see the mount line this script prints when it is not.  Ctrl+C stops both
and deletes the file.  Each server is the usual serve/server.py with the config setup wrote, so the API is the same.
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--config", default="strata-q2_0.json", help="the run config setup wrote (default: strata-q2_0.json here)")
ap.add_argument("--arena", default="/mnt/strata-shm/arena.bin", help="the shared arena file, on a tmpfs")
ap.add_argument("--gpu-a", default="0", help="HIP device of instance A, which pins the arena (the faster-prompt card)")
ap.add_argument("--gpu-b", default="1", help="HIP device of instance B, which only locks it")
ap.add_argument("--port-a", type=int, default=8080)
ap.add_argument("--port-b", type=int, default=8081)
ap.add_argument("--kv", default=None, help="fp16|int8|q4_0|k8v4 for both (default: the config's)")
ap.add_argument("--reserve-b", default="1500", help="--vram-reserve-mib of instance B (a 32 GB gfx906 card)")
ap.add_argument("--ring", default="512", help="STRATA_PREFILL_RING for both")
ap.add_argument("--min-arena-gib", type=float, default=34.0)
ap.add_argument("--keep-arena", action="store_true", help="leave the file in place on exit (the next start reloads it)")
ap.add_argument("--timeout", type=float, default=900)
a = ap.parse_args()

cfg0 = json.loads(Path(a.config).expanduser().read_text(encoding="utf-8-sig"))
arena = Path(a.arena)
arena.parent.mkdir(parents=True, exist_ok=True)
fs = subprocess.run(["stat", "-f", "-c", "%T", str(arena.parent)], capture_output=True, text=True).stdout.strip()
st = os.statvfs(arena.parent)
free_gib = st.f_bavail * st.f_frsize / 2**30
if fs != "tmpfs" or free_gib < a.min_arena_gib:
    sys.exit(f"{arena.parent} is {fs or 'unknown'} with {free_gib:.0f} GiB free: the arena needs a tmpfs of at least "
             f"{a.min_arena_gib:.0f} GiB.\n  sudo mkdir -p {arena.parent} && sudo mount -t tmpfs -o size=40G,mode=1777 tmpfs {arena.parent}")


def mkcfg(name, gpu, port, b):
    c = json.loads(json.dumps(cfg0))
    args = c["args"]
    if a.kv:
        args[args.index("--kv") + 1] = a.kv
    if b and "--vram-reserve-mib" in args:
        args[args.index("--vram-reserve-mib") + 1] = a.reserve_b
    args += ["--shared-expert-arena", str(arena)]
    c["args_written"] = args
    c["gpu"] = int(gpu)
    c.pop("layer_split", None)
    c.pop("gpus_asked", None)
    c.pop("vision", None)                      # the GPU picture reader would take VRAM beside the expert cache
    c["port"] = port
    c["log"] = str(Path(c.get("cwd", ROOT)) / f"strata-dual-{name}.log")
    c["env"] = {**c.get("env", {}), "STRATA_PREFILL_RING": a.ring, **({"STRATA_ARENA_PIN_GIB": "none"} if b else {})}
    p = Path(c.get("cwd", ROOT)) / f"strata-dual-{name}.json"
    p.write_text(json.dumps(c, indent=1))
    return p, c["log"]


procs = []


def stop(*_):
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except OSError:
            pass
    for p in procs:
        try:
            p.wait(60)
        except Exception:
            try: os.killpg(p.pid, signal.SIGKILL)
            except OSError: pass
    if not a.keep_arena:
        arena.unlink(missing_ok=True)
    sys.exit(0)


def start(name, gpu, port, b):
    cfg, log = mkcfg(name, gpu, port, b)
    p = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg),
                          "--port", str(port), "--gpu", str(gpu), "--no-api-key"], cwd=str(ROOT),
                         stdout=open(f"{log}.server", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(p)
    t0 = time.time()
    while True:
        if p.poll() is not None:
            stop_msg = f"instance {name} exited with {p.returncode}: see {log}.server and {log}"
            for q in procs:
                if q is not p:
                    os.killpg(q.pid, signal.SIGTERM)
            if not a.keep_arena:
                arena.unlink(missing_ok=True)
            sys.exit(stop_msg)
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(
                {"model": "x", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 2, "temperature": 0}).encode(),
                {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=30).read()
            break
        except Exception:
            if time.time() - t0 > a.timeout:
                stop()
            time.sleep(4)
    gn = subprocess.run(f"grep -hm1 'generate: GPU [0-9]:' {log}", shell=True, capture_output=True, text=True).stdout.strip()
    print(f"instance {name}: http://127.0.0.1:{port}  {gn.replace('strata generate: ', '')}  ({time.time() - t0:.0f} s)", flush=True)


signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
start("A", a.gpu_a, a.port_a, False)     # first: it fills the arena and pins it
start("B", a.gpu_b, a.port_b, True)      # second: maps the filled arena, locked but not pinned
print(f"\nboth servers are up.  Claude Code, one terminal per instance:\n"
      f"  ANTHROPIC_BASE_URL=http://127.0.0.1:{a.port_a} ANTHROPIC_AUTH_TOKEN=x ANTHROPIC_MODEL=claude-sonnet-4-5 claude\n"
      f"  ANTHROPIC_BASE_URL=http://127.0.0.1:{a.port_b} ANTHROPIC_AUTH_TOKEN=x ANTHROPIC_MODEL=claude-sonnet-4-5 claude\n"
      "Ctrl+C stops both.", flush=True)
while True:
    time.sleep(3600)
