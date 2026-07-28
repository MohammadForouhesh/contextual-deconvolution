#!/usr/bin/env python3
"""
Single-command runner for the Contextual Deconvolution paper artifacts.

Stages
------
  experiments   run every script in experiments/  (fits models, writes paper/results/*)
  figures       run every script in figures/       (renders paper/figures/*.png)
  reproduce     run the pytest suite in tests/      (determinism / sanity checks)
  all           experiments -> figures -> reproduce  (default)

Usage
-----
  python run_paper.py                 # = python run_paper.py all
  python run_paper.py experiments     # only the experiments
  python run_paper.py figures         # only the figures
  python run_paper.py reproduce       # only the tests
  python run_paper.py all --list      # dry run: show ordered plan, execute nothing
  python run_paper.py all --stop-on-error

Notes
-----
* Scripts are discovered automatically (sorted) from each folder; nothing here
  edits or imports them. PYTHONPATH is set to code/, code/experiments/ and
  code/figures/ so the handful of cross-folder sibling imports keep resolving.
* maintenance/ (paper .tex patchers + profiling utils) is intentionally NOT run.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(CODE_DIR, "experiments")
FIG_DIR = os.path.join(CODE_DIR, "figures")
TEST_DIR = os.path.join(CODE_DIR, "tests")


def _ensure_compat_symlinks():
    """Scripts moved into experiments/ and figures/ resolve some paths relative to
    their own location (dirname(__file__)/../data, /../paper). Recreate those anchors
    inside code/ as symlinks so the moved scripts find repo-root data/ and paper/.
    No script is edited; this only adds filesystem links."""
    for name, target in (("data", os.path.join("..", "data")),
                         ("paper", os.path.join("..", "paper"))):
        link = os.path.join(CODE_DIR, name)
        if os.path.islink(link) or os.path.exists(link):
            continue
        if os.path.isdir(os.path.join(CODE_DIR, target)):
            os.symlink(target, link)
            print(f"  (compat) linked code/{name} -> {target}")


def _scripts(folder):
    return sorted(glob.glob(os.path.join(folder, "*.py")))


def _env():
    env = os.environ.copy()
    extra = os.pathsep.join([CODE_DIR, EXP_DIR, FIG_DIR])
    env["PYTHONPATH"] = extra + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _rel(path):
    return os.path.relpath(path, CODE_DIR)


def run_stage(name, paths, results, stop_on_error, dry):
    n = len(paths)
    print("\n" + "=" * 72)
    print(f"STAGE: {name.upper()}  ({n} script{'s' if n != 1 else ''})")
    print("=" * 72, flush=True)
    for i, path in enumerate(paths, 1):
        tag = f"[{name} {i}/{n}]"
        if dry:
            print(f"  {tag} would run  {_rel(path)}")
            continue
        print(f"\n▶ {tag} {_rel(path)}", flush=True)
        t0 = time.time()
        rc = subprocess.call([sys.executable, path], cwd=CODE_DIR, env=_env())
        dt = time.time() - t0
        ok = rc == 0
        mark = "✓ OK " if ok else f"✗ FAILED (exit {rc})"
        print(f"{mark}  {_rel(path)}  [{dt:5.1f}s]", flush=True)
        results.append((_rel(path), ok, dt))
        if not ok and stop_on_error:
            print("\nStopping: --stop-on-error set and a script failed.", flush=True)
            return False
    return True


def run_reproduce(results, stop_on_error, dry):
    print("\n" + "=" * 72)
    print("STAGE: REPRODUCE  (pytest tests/)")
    print("=" * 72, flush=True)
    if dry:
        print(f"  [reproduce 1/1] would run  pytest {_rel(TEST_DIR)}")
        return True
    print(f"\n▶ [reproduce 1/1] pytest {_rel(TEST_DIR)}", flush=True)
    t0 = time.time()
    rc = subprocess.call([sys.executable, "-m", "pytest", "-q", TEST_DIR], cwd=CODE_DIR, env=_env())
    dt = time.time() - t0
    ok = rc == 0
    print(f"{'✓ OK ' if ok else f'✗ FAILED (exit {rc})'}  pytest  [{dt:5.1f}s]", flush=True)
    results.append(("pytest tests/", ok, dt))
    return ok or not stop_on_error


def main():
    ap = argparse.ArgumentParser(description="Run paper experiments, figures, and reproducibility checks.")
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["all", "experiments", "figures", "reproduce"])
    ap.add_argument("--list", action="store_true", help="dry run: print the ordered plan, execute nothing")
    ap.add_argument("--stop-on-error", action="store_true", help="halt at the first failing script")
    args = ap.parse_args()

    _ensure_compat_symlinks()
    exp, fig = _scripts(EXP_DIR), _scripts(FIG_DIR)
    results = []
    t0 = time.time()

    do_exp = args.stage in ("all", "experiments")
    do_fig = args.stage in ("all", "figures")
    do_rep = args.stage in ("all", "reproduce")

    cont = True
    if do_exp and cont:
        cont = run_stage("experiments", exp, results, args.stop_on_error, args.list)
    if do_fig and cont:
        cont = run_stage("figures", fig, results, args.stop_on_error, args.list)
    if do_rep and cont:
        run_reproduce(results, args.stop_on_error, args.list)

    if args.list:
        return
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    n_ok = sum(1 for _, ok, _ in results if ok)
    for name, ok, dt in results:
        print(f"  {'✓' if ok else '✗'}  {name:<48s} {dt:6.1f}s")
    print("-" * 72)
    print(f"  {n_ok}/{len(results)} succeeded   total {time.time() - t0:6.1f}s")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
