
from __future__ import annotations
import argparse, glob, time
from pathlib import Path
from inverover import run_on_instance

def main():
    ap = argparse.ArgumentParser(
        description="Run Inver-over EA (Tao & Michalewicz) on TSPLIB instances and write results/inverover.txt"
    )
    ap.add_argument("--instances_glob", type=str, required=True,
                    help='Glob for .tsp files, e.g. "tsplib/*.tsp"')
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--gens", type=int, default=20000)
    ap.add_argument("--p_random", type=float, default=0.02)
    ap.add_argument("--out", type=str, default="results/inverover.txt")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(args.instances_glob))
    if not files:
        raise SystemExit(f"No instances matched: {args.instances_glob}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Inver-over EA results (mean, stddev over runs)\n")
        f.write(f"# runs={args.runs}, pop={args.pop}, gens={args.gens}, p_random={args.p_random}\n")
        f.write("instance,mean_cost,stddev\n")
        for p in files:
            name = Path(p).name
            print(f"=== {name}: runs={args.runs}, pop={args.pop}, gens={args.gens}, p={args.p_random} ===", flush=True)
            t0 = time.time()
            mean, std = run_on_instance(
                p, runs=args.runs, population_size=args.pop,
                generations=args.gens, p_random=args.p_random,
                seed=args.seed, progress_every=max(args.gens // 10, 1)
            )
            dt = time.time() - t0
            f.write(f"{name},{mean:.6f},{std:.6f}\n")
            f.flush()
            print(f"[DONE] {name}: mean={mean:.2f} std={std:.2f} ({dt:.1f}s)", flush=True)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
