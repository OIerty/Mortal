"""
run_pipeline.py – Iterative self-play reinforcement learning pipeline.

This script automates the full RL training loop:

    Bootstrap
    ─────────
    1. Generate a small synthetic dataset (if no seed checkpoint exists).
    2. Train an initial model for ``--bootstrap-epochs`` epochs.

    RL Iteration (repeat N times)
    ─────────────────────────────
    3. Run ``--games-per-iter`` self-play games using the current checkpoint.
    4. Merge new data with a rolling window of recent data
       (controlled by ``--replay-games``).
    5. Fine-tune the model for ``--epochs-per-iter`` epochs on the merged data.
    6. Evaluate offline metrics; log progress.
    7. Save checkpoint for the next iteration.

Usage
-----
    # Quick smoke test (CPU, synthetic engine)
    python run_pipeline.py --iterations 3 --games-per-iter 20 --epochs-per-iter 1

    # Production run (GPU)
    python run_pipeline.py \\
        --iterations 50 \\
        --games-per-iter 500 \\
        --epochs-per-iter 5 \\
        --batch-size 512 \\
        --conv-channels 192 \\
        --num-blocks 40 \\
        --device cuda \\
        --out-dir pipeline_out

    # Resume a previous pipeline
    python run_pipeline.py \\
        --resume pipeline_out/iter_010/checkpoint.pth \\
        --out-dir pipeline_out \\
        --iterations 100

Notes
-----
* For realistic training, provide a real mahjong engine via --engine-module.
  See selfplay.py for the expected interface.
* The pipeline keeps the last ``--replay-games`` games worth of data in a
  rolling JSONL buffer to prevent catastrophic forgetting.
* Checkpoints and data for every iteration are saved under --out-dir so
  training can be resumed at any point.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: Path | None = None) -> int:
    """Run a subprocess command, streaming output to stdout/stderr."""
    logger.debug("$ %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _tail_jsonl(src: Path, dst: Path, keep_lines: int):
    """Write the last *keep_lines* lines of src into dst (streaming, O(keep_lines) memory)."""
    if not src.exists():
        return
    from collections import deque
    with src.open("r", encoding="utf-8") as f:
        tail: deque[str] = deque(f, maxlen=keep_lines)
    with dst.open("w", encoding="utf-8") as f:
        f.writelines(tail)


def _merge_jsonl(files: list[Path], dst: Path, keep_last: int | None = None):
    """Concatenate JSONL files into dst, optionally keeping only the last N lines.

    Uses unique temporary files (via :mod:`tempfile`) so that:
    * concurrent runs targeting the same *dst* do not collide on fixed names, and
    * if *dst* is also in *files* (rolling-buffer pattern) it is not truncated
      before its contents are read.
    Both temp files are removed in a ``finally`` block to avoid leaks on error.
    """
    parent = dst.parent
    merge_tmp_path: Path | None = None
    tail_tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=parent, suffix=".merge_tmp", delete=False
        ) as merge_tmp:
            merge_tmp_path = Path(merge_tmp.name)
            for src in files:
                if src.exists():
                    with src.open("r", encoding="utf-8") as inp:
                        shutil.copyfileobj(inp, merge_tmp)

        if keep_last is not None:
            with tempfile.NamedTemporaryFile(
                dir=parent, suffix=".tail_tmp", delete=False
            ) as tail_tmp:
                tail_tmp_path = Path(tail_tmp.name)
            _tail_jsonl(merge_tmp_path, tail_tmp_path, keep_last)
            tail_tmp_path.replace(dst)
            tail_tmp_path = None  # replaced successfully; no cleanup needed
        else:
            merge_tmp_path.replace(dst)
            merge_tmp_path = None  # replaced successfully; no cleanup needed
    finally:
        if merge_tmp_path is not None and merge_tmp_path.exists():
            merge_tmp_path.unlink()
        if tail_tmp_path is not None and tail_tmp_path.exists():
            tail_tmp_path.unlink()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).parent
    python = sys.executable

    # ------------------------------------------------------------------ Bootstrap
    checkpoint_path = Path(args.resume) if args.resume else None

    if checkpoint_path is None:
        logger.info("=" * 60)
        logger.info("BOOTSTRAP: generating synthetic seed data and training initial model")
        logger.info("=" * 60)

        seed_data = out_dir / "seed.jsonl"
        rc = _run([
            python, str(script_dir / "generate_sample.py"),
            "--n", str(args.bootstrap_samples),
            "--out", str(seed_data),
        ])
        if rc != 0:
            logger.error("generate_sample.py failed (rc=%d)", rc)
            sys.exit(rc)

        checkpoint_path = out_dir / "bootstrap.pth"
        train_cmd = _build_train_cmd(
            python, script_dir, seed_data, checkpoint_path, args,
            epochs=args.bootstrap_epochs,
        )
        rc = _run(train_cmd)
        if rc != 0:
            logger.error("Bootstrap training failed (rc=%d)", rc)
            sys.exit(rc)
        logger.info("Bootstrap checkpoint: %s", checkpoint_path)
    else:
        logger.info("Resuming from checkpoint: %s", checkpoint_path)

    # ------------------------------------------------------------------ Rolling replay buffer
    replay_path = out_dir / "replay.jsonl"
    examples_per_game = args.steps_per_game * 3  # 3 seats × steps_per_game transitions
    replay_lines = args.replay_games * examples_per_game

    # ------------------------------------------------------------------ RL iterations
    start_iter = args.start_iter
    for iteration in range(start_iter, start_iter + args.iterations):
        iter_dir = out_dir / f"iter_{iteration:04d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        logger.info("=" * 60)
        logger.info("ITERATION %d / %d", iteration, start_iter + args.iterations - 1)
        logger.info("=" * 60)

        # Step 1: Self-play
        selfplay_data = iter_dir / "selfplay.jsonl"
        selfplay_cmd = [
            python, str(script_dir / "selfplay.py"),
            "--checkpoint", str(checkpoint_path),
            "--games", str(args.games_per_iter),
            "--out", str(selfplay_data),
            "--epsilon", str(args.epsilon),
            "--temperature", str(args.temperature),
            "--device", args.device,
            "--steps-per-game", str(args.steps_per_game),
        ]
        if args.random_seats:
            selfplay_cmd += ["--random-seats"] + [str(s) for s in args.random_seats]
        if args.engine_module:
            selfplay_cmd += ["--engine-module", args.engine_module]

        logger.info("Step 1/3: self-play (%d games)…", args.games_per_iter)
        rc = _run(selfplay_cmd)
        if rc != 0:
            logger.error("selfplay.py failed (rc=%d), skipping iteration", rc)
            continue

        n_sp = _count_lines(selfplay_data)
        logger.info("  Collected %d examples from self-play", n_sp)

        # Step 2: Merge new data into replay buffer
        logger.info("Step 2/3: updating replay buffer (keep last %d lines)…", replay_lines)
        _merge_jsonl([replay_path, selfplay_data], replay_path, keep_last=replay_lines)
        n_replay = _count_lines(replay_path)
        logger.info("  Replay buffer: %d examples", n_replay)

        # Step 3: Fine-tune
        new_checkpoint = iter_dir / "checkpoint.pth"
        train_cmd = _build_train_cmd(
            python, script_dir, replay_path, new_checkpoint, args,
            epochs=args.epochs_per_iter,
            resume=checkpoint_path,
        )
        logger.info("Step 3/3: fine-tuning for %d epoch(s)…", args.epochs_per_iter)
        rc = _run(train_cmd)
        if rc != 0:
            logger.error("train.py failed (rc=%d), keeping old checkpoint", rc)
            continue

        checkpoint_path = new_checkpoint

        # Optional offline eval
        if args.eval_data and Path(args.eval_data).exists():
            eval_out = iter_dir / "eval.json"
            rc = _run([
                python, str(script_dir / "eval.py"),
                "--checkpoint", str(checkpoint_path),
                "--data", args.eval_data,
                "--device", args.device,
                "--out-json", str(eval_out),
            ])
            if rc == 0 and eval_out.exists():
                metrics = json.loads(eval_out.read_text())
                offline = metrics.get("offline", {})
                logger.info(
                    "  Eval: policy_acc=%.4f  top5_acc=%.4f  value_mae=%.4f",
                    offline.get("policy_acc", 0),
                    offline.get("top5_acc", 0),
                    offline.get("value_mae", 0),
                )

        elapsed = time.time() - t0
        logger.info("Iteration %d done in %.1fs  →  %s", iteration, elapsed, checkpoint_path)

    # ------------------------------------------------------------------ Final
    final_path = out_dir / "mortal_3p_final.pth"
    if checkpoint_path and checkpoint_path.exists():
        shutil.copy2(checkpoint_path, final_path)
        logger.info("=" * 60)
        logger.info("Pipeline complete.")
        logger.info("Final checkpoint: %s", final_path)
        logger.info("Copy to Akagi:  cp %s /path/to/Akagi/mjai_bot/mortal3p/mortal.pth", final_path)
    else:
        logger.warning("No checkpoint was produced.")


def _build_train_cmd(
    python: str,
    script_dir: Path,
    data: Path,
    out: Path,
    args: argparse.Namespace,
    epochs: int,
    resume: Path | None = None,
) -> list[str]:
    cmd = [
        python, str(script_dir / "train.py"),
        "--data", str(data),
        "--epochs", str(epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--conv-channels", str(args.conv_channels),
        "--num-blocks", str(args.num_blocks),
        "--device", args.device,
        "--out", str(out),
        "--cql-weight", str(args.cql_weight),
        "--num-workers", str(args.num_workers),
    ]
    if resume and resume.exists():
        cmd += ["--resume", str(resume)]
    if args.enable_amp:
        cmd.append("--enable-amp")
    return cmd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterative self-play RL training pipeline for 3P mahjong",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Pipeline control
    pipeline = parser.add_argument_group("Pipeline")
    pipeline.add_argument("--iterations", type=int, default=10,
                          help="Number of RL iterations to run")
    pipeline.add_argument("--start-iter", type=int, default=0,
                          help="Iteration index to start from (useful when resuming)")
    pipeline.add_argument("--out-dir", default="pipeline_out",
                          help="Root output directory for checkpoints and data")
    pipeline.add_argument("--resume", default="",
                          help="Path to an existing checkpoint to resume from "
                               "(skips bootstrap if provided)")

    # Bootstrap
    bootstrap = parser.add_argument_group("Bootstrap")
    bootstrap.add_argument("--bootstrap-samples", type=int, default=200,
                           help="Synthetic examples for the bootstrap phase")
    bootstrap.add_argument("--bootstrap-epochs", type=int, default=3,
                           help="Training epochs for the bootstrap phase")

    # Self-play
    sp = parser.add_argument_group("Self-play")
    sp.add_argument("--games-per-iter", type=int, default=100,
                    help="Self-play games per RL iteration")
    sp.add_argument("--steps-per-game", type=int, default=40,
                    help="Max steps per game (built-in synthetic engine)")
    sp.add_argument("--replay-games", type=int, default=500,
                    help="Number of recent games to keep in the replay buffer")
    sp.add_argument("--epsilon", type=float, default=0.05,
                    help="ε-greedy exploration rate for model seats")
    sp.add_argument("--temperature", type=float, default=0.0,
                    help="Softmax temperature for action sampling (0=argmax)")
    sp.add_argument("--random-seats", type=int, nargs="*", default=[],
                    help="Seat indices that always play randomly")
    sp.add_argument("--engine-module", default="",
                    help="Python module.ClassName for a custom game engine")

    # Training
    training = parser.add_argument_group("Training")
    training.add_argument("--epochs-per-iter", type=int, default=3,
                          help="Fine-tuning epochs per RL iteration")
    training.add_argument("--batch-size", type=int, default=256)
    training.add_argument("--lr", type=float, default=5e-4)
    training.add_argument("--conv-channels", type=int, default=64)
    training.add_argument("--num-blocks", type=int, default=6)
    training.add_argument("--cql-weight", type=float, default=1.0,
                          help="CQL regularisation weight (helps offline RL stability)")
    training.add_argument("--num-workers", type=int, default=0)
    training.add_argument("--enable-amp", action="store_true")
    training.add_argument("--device", default="cpu")

    # Evaluation
    ev = parser.add_argument_group("Evaluation")
    ev.add_argument("--eval-data", default="",
                    help="JSONL file to use for offline evaluation after each iteration")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    t0 = time.time()
    run_pipeline(args)
    logger.info("Total pipeline time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(0)
