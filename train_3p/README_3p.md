# 三麻 (Three-Player Mahjong) Model Training Pipeline

This directory contains a reproducible, self-contained training pipeline for a
three-player mahjong (三麻 / 3P) AI model that is compatible with the
[Akagi](https://github.com/OIerty/Akagi) / MJAI ecosystem.

---

## Quick Start (Smoke Test)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic sample data (20 examples)
python generate_sample.py

# 3. Train a baseline model for 1 epoch
python train.py --data sample.jsonl --epochs 1

# 4. Verify the checkpoint was created
ls -lh mortal_3p.pth
```

After these steps you will have `mortal_3p.pth` ready to integrate with Akagi.

---

## Directory Structure

```
train_3p/
├── data_converter.py       # Convert mahjong logs → training JSONL/NPZ
├── dataset.py              # PyTorch Dataset (loads JSONL / NPZ / PT)
├── model.py                # Brain + DQN model (Mortal-compatible checkpoint)
├── train.py                # Supervised training script
├── eval.py                 # Offline + self-play evaluation
├── generate_sample.py      # Synthetic data generator for smoke tests
├── requirements.txt        # Python dependencies
├── settings_3p_example.json # Example Akagi settings.json for 3P
├── mjai_adapter_patch.diff # Patch guide for Akagi integration
├── tests/
│   ├── test_data_converter.py   # Unit tests for data_converter.py
│   └── test_dataset.py          # Unit tests for dataset.py + model.py
└── README_3p.md            # This file
```

---

## Step 1: Prepare Training Data

### Option A – From MJAI JSON Logs

MJAI logs are JSON files (or JSONL) containing a list of game events.
Sources: [smly/mjai.app](https://github.com/smly/mjai.app),
hand-collected self-play logs, or any MJAI-compatible tool.

```bash
# Single file
python data_converter.py --input game.json --fmt mjai --output train.jsonl

# Multiple files (use a shell loop or glob)
for f in logs/*.json; do
    python data_converter.py --input "$f" --fmt mjai --output tmp.jsonl
    cat tmp.jsonl >> train.jsonl
done
```

### Option B – From mahjong_soul_api / Mahjong Soul Logs

The [mahjong_soul_api](https://github.com/OIerty/mahjong_soul_api) library
lets you download game records from Mahjong Soul.

```bash
# Download logs using mahjong_soul_api (see that repo for setup instructions)
# Then convert the resulting JSON files:
python data_converter.py --input majsoul_game.json --fmt majsoul --output train.jsonl
```

> **Manual step**: Mahjong Soul log downloads require authentication.
> Follow the [mahjong_soul_api README](https://github.com/OIerty/mahjong_soul_api)
> to set up credentials and download `*.json` game records.
> Do **not** commit credentials to this repository.

### Option C – From Bot Action Logs

If you already have a bot that produces action logs (e.g. from a bot.zip),
each line must be a JSON object:

```json
{"state": [0.0, ...], "action": 5, "legal": [true, false, ...], "reward": 1000.0}
```

```bash
python data_converter.py --input bot.jsonl --fmt bot --output train.jsonl
```

### Output Format

Each line in the output JSONL has:

```json
{
  "state": [float, ...],              // OBS_CHANNELS_3P × 34 = 1836 floats
  "legal_actions_mask": [bool, ...],  // 79 booleans
  "action": 5,                        // action index (see action table below)
  "reward": 1000.0,                   // final score delta
  "meta": {
    "game_id": "...",
    "player_id": 0,
    "round": "E1",
    "step": 42,
    "is_3p": true
  }
}
```

---

## Step 2: Action Index Table

The 3P action space has **79** distinct actions:

| Index  | MJAI Type   | Description                                        |
|--------|-------------|----------------------------------------------------|
| 0–33   | `dahai`     | Discard tile (tile_id = index, see Tile Encoding)  |
| 34–67  | `dahai` + riichi | Riichi declaration + discard (tile_id = index−34) |
| 68     | `agari`     | Win (covers tsumo AND ron)                         |
| 69     | `pon`       | Pon call on last discard                           |
| 70     | `chi_low`   | Chi: own tiles are below the called tile           |
| 71     | `chi_mid`   | Chi: own tiles sandwich the called tile            |
| 72     | `chi_high`  | Chi: own tiles are above the called tile           |
| 73     | `daiminkan` | Open kan on opponent discard                       |
| 74     | `kakan`     | Added kan (shouminkan)                             |
| 75     | `ankan`     | Closed kan                                         |
| 76     | `ryukyoku`  | Abortive/exhaustive draw                           |
| 77     | `none`      | Pass / no action                                   |
| 78     | `nukidora`  | **3P-only** – declare North tile as extra dora     |

### Tile Encoding (tile_id 0–33)

| tile_id | Tile  | tile_id | Tile  | tile_id | Tile  |
|---------|-------|---------|-------|---------|-------|
| 0–8     | 1m–9m | 9–17    | 1p–9p | 18–26   | 1s–9s |
| 27      | 1z (E)| 28      | 2z (S)| 29      | 3z (W)|
| 30      | 4z (N)| 31      | 5z (白)| 32     | 6z (發)|
| 33      | 7z (中)|

> In 3P mahjong, **North tiles (4z / tile_id=30)** are removed from the draw
> pile and declared as `nukidora` instead. The training pipeline encodes this
> with action index **78**.

### 3P vs 4P Differences

| Feature          | 4P (Standard)   | 3P (Sanma)                        |
|------------------|-----------------|-----------------------------------|
| Players          | 4               | 3                                 |
| Wall tiles       | 136             | ~108 (North tiles removed)        |
| Seat winds       | E / S / W / N   | E / S / W (no North seat)         |
| Starting score   | 25000           | 35000 (common rule set)           |
| North tile       | Normal discard  | Nukidora (extra dora declaration) |
| Chi              | Downstream only | Same rules apply                  |

---

## Step 3: Train the Model

### Basic Training

```bash
python train.py --data train.jsonl --epochs 10
```

### With Validation and GPU

```bash
python train.py \
    --data  train.jsonl \
    --val   val.jsonl \
    --epochs 30 \
    --batch-size 512 \
    --lr 1e-3 \
    --conv-channels 192 \
    --num-blocks 40 \
    --device cuda \
    --out mortal_3p.pth
```

### Architecture Options

| Flag              | Default | Production | Description                  |
|-------------------|---------|------------|------------------------------|
| `--conv-channels` | 64      | 192        | ResNet filter count          |
| `--num-blocks`    | 6       | 40         | ResNet residual blocks       |
| `--model-version` | 4       | 4          | Architecture version (2–4)   |
| `--batch-size`    | 256     | 512        | Training batch size          |
| `--lr`            | 1e-3    | 1e-4       | Peak learning rate           |
| `--cql-weight`    | 0.0     | 5.0        | CQL regularisation (offline) |

**Checkpoint naming:** The final checkpoint is saved to `--out` (default:
`mortal_3p.pth`). The best checkpoint by validation accuracy is saved to
`best_3p.pth` in the same directory.

### Resuming Training

```bash
python train.py --data train.jsonl --resume mortal_3p.pth --epochs 60
```

---

## Step 4: Evaluate the Model

### Offline Evaluation

```bash
python eval.py --checkpoint mortal_3p.pth --data val.jsonl
```

Outputs:

- **Policy accuracy** – fraction of decisions matching human/bot actions
- **Top-5 accuracy** – correct action in top-5 Q-values
- **Value MAE** – mean absolute error between Q(s,a) and observed score delta

### Self-play vs Random Baseline

```bash
python eval.py --checkpoint mortal_3p.pth --self-play --games 20
```

> **Note**: The built-in self-play harness uses a synthetic random environment.
> For realistic strength measurement, integrate with a full mahjong engine
> (e.g. use Akagi's test-play mode after deployment).

### Save Results to JSON

```bash
python eval.py --checkpoint mortal_3p.pth --data val.jsonl --out-json results.json
```

---

## Step 5: Export and Integrate with Akagi (shinkuan/Akagi v2)

The checkpoint produced by `train.py` (`mortal_3p.pth`) is **drop-in compatible**
with the `mjai_bot/mortal3p` bot in
[shinkuan/Akagi v2](https://github.com/shinkuan/Akagi/tree/v2/mjai_bot/mortal3p).
Just replace the pre-trained `mortal.pth` file with your own — no code changes
required.  The loader in `mjai_bot/mortal3p/model.py` reads the architecture
parameters (`version`, `conv_channels`, `num_blocks`) directly from the
checkpoint's `config` dict, so any model size works.

### Checkpoint Format

The saved checkpoint (`mortal_3p.pth`) contains:

```python
{
    "mortal":       mortal.state_dict(),   # Brain (ResNet encoder) weights
    "current_dqn":  dqn.state_dict(),      # DQN (policy/value head) weights
    "steps":        int,                   # training step count
    "timestamp":    float,                 # Unix timestamp of save
    "config": {
        "control": {"version": int},       # model version (2/3/4)
        "resnet":  {"conv_channels": int,
                    "num_blocks":    int},
        "is_3p":   True,
    },
}
```

This matches exactly what `shinkuan/Akagi v2 mjai_bot/mortal3p/model.py`
expects when calling `torch.load("mortal.pth", ...)`.

### Weight File Placement

```
Akagi/
└── mjai_bot/
    └── mortal3p/
        ├── bot.py          ← unchanged
        ├── model.py        ← unchanged (reads config from checkpoint)
        ├── mortal.pth      ← REPLACE with your mortal_3p.pth
        └── libriichi3p.pyd / libriichi3p.so
```

```bash
# Windows
copy mortal_3p.pth  C:\path\to\Akagi\mjai_bot\mortal3p\mortal.pth

# macOS / Linux
cp mortal_3p.pth  /path/to/Akagi/mjai_bot/mortal3p/mortal.pth
```

> **Tip**: The default model trained by `generate_sample.py + train.py` is very
> small (64 channels × 6 blocks, ~0.5 M params) and serves as a smoke-test
> baseline.  For a competitive model increase `--conv-channels 192 --num-blocks 40`
> and train on real game records.

### Update settings.json

Edit `Akagi/settings/settings.json` and change the `"model"` field:

```json
{
  "model": "mortal3p",
  ...
}
```

See `settings_3p_example.json` in this directory for a complete example.

### Windows Installation Flow

1. Install Python 3.10–3.12 and `pip install -r requirements.txt`.
2. Follow the
   [Akagi README](https://github.com/shinkuan/Akagi/blob/v2/README.md)
   to install Akagi dependencies.
3. Copy `mortal_3p.pth` → `Akagi\mjai_bot\mortal3p\mortal.pth`.
4. Edit `Akagi\settings\settings.json` → `"model": "mortal3p"`.
5. Open Mahjong Soul in your browser, navigate to a three-player room.
6. Run `python run_akagi.py` (or the provided batch file).

### macOS Installation Flow

1. Install Python 3.10–3.12 and `pip install -r requirements.txt`.
2. Follow the Akagi macOS instructions (requires Homebrew + mitmproxy certs).
3. Copy `mortal_3p.pth` → `Akagi/mjai_bot/mortal3p/mortal.pth`.
4. Edit `Akagi/settings/settings.json` → `"model": "mortal3p"`.
5. Run `python run_akagi.py`.

### Adapter Patch

See `mjai_adapter_patch.diff` for:

- Optional improvements to `mjai_bot/mortal3p/model.py` to dynamically read
  architecture parameters from the checkpoint config.
- An action-space guard that warns if `libriichi3p.ACTION_SPACE != 79`.
- A copy-pasteable smoke test to verify the bot loads and acts correctly.

---

## Step 6: Self-Play Reinforcement Learning

Once you have a supervised baseline model, you can improve it further through
self-play RL using the iterative pipeline.

### How it works

```
Bootstrap (once)
  └─ generate_sample.py  →  seed.jsonl  →  train.py  →  bootstrap.pth

RL Loop (N iterations)
  └─ selfplay.py  →  new examples  →  merge into replay buffer
  └─ train.py     →  fine-tune on replay buffer  →  checkpoint_iter_N.pth
  └─ eval.py      →  offline metrics per iteration
```

### Quick start (CPU smoke test)

```bash
cd train_3p
python run_pipeline.py --iterations 3 --games-per-iter 20 --epochs-per-iter 1
```

This bootstraps a tiny model, runs 3 iterations of self-play + training, and
writes all outputs to `pipeline_out/`.

### Production run (GPU recommended)

```bash
python run_pipeline.py \
    --iterations 50 \
    --games-per-iter 500 \
    --epochs-per-iter 5 \
    --batch-size 512 \
    --conv-channels 192 \
    --num-blocks 40 \
    --device cuda \
    --out-dir pipeline_out
```

### Resume a paused pipeline

```bash
python run_pipeline.py \
    --resume pipeline_out/iter_0010/checkpoint.pth \
    --start-iter 11 \
    --iterations 50 \
    --out-dir pipeline_out
```

### Self-play standalone

Generate self-play data independently (useful for data collection on a
separate machine):

```bash
# All seats use the model (ε=0.05 greedy)
python selfplay.py --checkpoint mortal_3p.pth --games 200 --out sp200.jsonl

# Seat 2 plays randomly (more exploration diversity)
python selfplay.py --checkpoint mortal_3p.pth --games 200 --random-seats 2 --out sp200.jsonl

# Softmax temperature sampling (more varied actions)
python selfplay.py --checkpoint mortal_3p.pth --games 200 --temperature 0.5 --out sp200.jsonl
```

### Plug in a real mahjong engine

The built-in engine generates random states and random outcomes – enough to
test the pipeline, but the model won't learn real mahjong strategy from it.

To use a real engine, implement the two-method interface in `selfplay.py`:

```python
class MahjongEngine:
    def reset(self):
        # Start a new game; return (states, masks) for each of 3 seats
        ...

    def step(self, actions):
        # Apply actions; return (next_states, next_masks, rewards, done)
        ...
```

Then pass it to the scripts:

```bash
python selfplay.py \
    --checkpoint mortal_3p.pth \
    --engine-module mypackage.myengine.MahjongEngine \
    --games 500 --out sp500.jsonl

python run_pipeline.py \
    --engine-module mypackage.myengine.MahjongEngine \
    --iterations 50 ...
```

### Key pipeline parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--iterations` | 10 | Number of RL iterations |
| `--games-per-iter` | 100 | Self-play games per iteration |
| `--epochs-per-iter` | 3 | Fine-tuning epochs per iteration |
| `--replay-games` | 500 | Rolling replay buffer size (in games) |
| `--epsilon` | 0.05 | ε-greedy exploration rate |
| `--temperature` | 0.0 | Softmax temperature (0 = argmax) |
| `--cql-weight` | 1.0 | Conservative Q-Learning penalty |
| `--bootstrap-epochs` | 3 | Epochs for the initial supervised boot |

---

## Step 7: Train on GitHub Codespaces (Cloud)

The repository includes a `.devcontainer/devcontainer.json` so you can train
directly in a browser-based cloud environment.

### What Codespaces can do

| Tier | CPU | RAM | GPU | Use case |
|------|-----|-----|-----|----------|
| Free (2-core) | 2 vCPU | 8 GB | ❌ | Smoke test, 1–2 epochs, tiny model |
| 4-core (paid) | 4 vCPU | 16 GB | ❌ | Short supervised runs, small model |
| GPU (paid) | 4 vCPU | 16 GB | T4 16 GB | Full training |

**Verdict**: Codespaces is great for testing the pipeline and short
supervised runs. For meaningful RL training (50+ iterations, 192-channel
model), you need a GPU – either a paid Codespaces machine type with GPU, or
a separate cloud VM (Colab, Vast.ai, RunPod, etc.).

### Launch Codespaces

1. Go to the repository on GitHub.
2. Click **Code → Codespaces → Create codespace on this branch**.
3. Wait ~2 minutes for the environment to build (Python 3.11 + dependencies
   are installed automatically via `postCreateCommand`).

### Run training in Codespaces

```bash
# In the Codespaces terminal:
cd train_3p

# Smoke test (CPU, ~30 seconds)
python run_pipeline.py --iterations 2 --games-per-iter 20 --epochs-per-iter 1

# Supervised baseline on your own data
python train.py --data /path/to/train.jsonl --epochs 5 --device cpu
```

### TensorBoard in Codespaces

TensorBoard port (6006) is forwarded automatically.  After training starts:

1. Click the **Ports** tab in VS Code (bottom panel).
2. Click the 🌐 icon next to port 6006.

### Upload real game data to Codespaces

```bash
# Option A: drag-and-drop in VS Code's Explorer panel

# Option B: use GitHub CLI
gh codespace cp ./mylogs/*.json remote:/workspaces/Mortal/train_3p/

# Option C: wget / curl from a URL
wget -P train_3p/ https://example.com/mahjong_data.jsonl
```

### Download the trained checkpoint

```bash
# From your local machine:
gh codespace cp remote:/workspaces/Mortal/train_3p/mortal_3p.pth ./mortal_3p.pth
```

---

## Step 8: Run Tests

```bash
# From the train_3p/ directory
pip install pytest
pytest tests/ -v
```

The test suite covers:

- Tile encoding helpers (all 34 tile types, red fives, roundtrips)
- Action mapping table (all 79 actions, inverse mapping)
- State encoder (shape, dtype, channel semantics)
- Legal actions mask generator
- GameState3P event application (tsumo, dahai, riichi, nukidora, …)
- MJAI log converter (3P games processed, 4P games skipped)
- Bot log converter
- JSONL / NPZ writer utilities
- MahjongDataset3P (JSONL, NPZ, max_samples, augmentation)
- Brain + DQN model shapes, masking, eval mode
- Checkpoint save / load roundtrip
- End-to-end: generate_sample → train → eval

---

## Observation State Layout (OBS_CHANNELS_3P = 54)

The state tensor has shape `(54, 34)`.  Each row is a feature channel over
the 34 tile types:

| Channel(s) | Content                                              |
|------------|------------------------------------------------------|
| 0–3        | Own hand binary counts (has ≥1, ≥2, ≥3, ≥4 copies)  |
| 4–7        | Own discard history (4 temporal buckets)              |
| 8–11       | Opponent-1 discard history                           |
| 12–15      | Opponent-2 discard history                           |
| 16–18      | Own melds (up to 3 active melds)                     |
| 19–21      | Opponent-1 melds                                     |
| 22–24      | Opponent-2 melds                                     |
| 25–29      | Dora indicators (up to 5 slots, one-hot)             |
| 30         | Remaining tiles (normalized by 70; broadcast)        |
| 31–33      | Riichi/tenpai state (self, opp1, opp2; broadcast)    |
| 34–36      | Score (self, opp1, opp2; normalized by 25000)        |
| 37         | Round wind (0=E, 1=S; broadcast)                     |
| 38         | Seat wind (0–2; broadcast)                           |
| 39         | Kyoku/honba/kyotaku info (broadcast)                 |
| 40–42      | Nukidora count (self, opp1, opp2; broadcast)         |
| 43         | Last discarded tile (one-hot over 34 positions)      |
| 44         | Current player (broadcast)                           |
| 45–53      | Reserved (zeros)                                     |

---

## Checkpoint Format

The produced `mortal_3p.pth` is a PyTorch state dict with the following keys:

```python
{
    "mortal":      <Brain state_dict>,   # state encoder weights
    "current_dqn": <DQN state_dict>,     # policy + value head weights
    "steps":       int,                  # training steps at save time
    "timestamp":   float,                # Unix timestamp
    "config": {
        "control": {"version": 4},
        "resnet":  {"conv_channels": 64, "num_blocks": 6},
        "is_3p":   True,                 # marks this as a 3P checkpoint
    },
}
```

This format is **directly compatible** with the loader in
`OIerty/Akagi/mjai_bot/mortal3p/model.py`.

---

## References

- [OIerty/Akagi](https://github.com/OIerty/Akagi) – Mahjong Soul AI assistant
- [OIerty/Mortal](https://github.com/OIerty/Mortal) – Original Mortal model (this repo)
- [OIerty/mjai.app](https://github.com/OIerty/mjai.app) – MJAI bot protocol
- [OIerty/mahjong_soul_api](https://github.com/OIerty/mahjong_soul_api) – Mahjong Soul API
- [OIerty/mahjong-helper](https://github.com/OIerty/mahjong-helper) – Mahjong rule reference
- [OIerty/majsoul_mod_plus](https://github.com/OIerty/majsoul_mod_plus) – Client mods
- [Equim-chan/Mortal](https://github.com/Equim-chan/Mortal) – Upstream Mortal model
- [smly/mjai.app](https://github.com/smly/mjai.app) – MJAI bot framework
