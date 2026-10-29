# ---
# jupyter:
#   colab:
#     name: snake.ipynb
#     provenance: []
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: colab,kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Teach an AI to play Snake
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eurecom-ds/intro-ai-challenges/blob/main/snake/notebook.ipynb)
#
# ![Teach an AI to play Snake. 60 minutes. Colab CPU. Learn through rewards.](https://raw.githubusercontent.com/eurecom-ds/intro-ai-challenges/5f2d202e703da1308a3472f23761362980a26159/snake/assets/overview.svg)
#
# How would you reward a snake for playing well? Food, survival and movement toward food each suggest a possible reward. Your choices will guide an agent as it learns from complete games.
#
# Train a **stochastic policy**, a neural network that assigns a probability to each action. Change the reward values through the supplied controls, predict the effect, then watch what the agent learns.
#
# The server ranks your selected agent by **food collected**, then **collection speed** when food totals tie. Everyone uses the same evaluation rules, whatever rewards they chose during training.
#
# | Minutes | Activity |
# |---|---|
# | 0–8 | Connect a CPU, inspect the game and watch an untrained agent. |
# | 8–15 | Train the starting agent and prepare Upload 1. |
# | 15–22 | Compare short and longer training. |
# | 22–34 | Investigate food and collision rewards. |
# | 34–46 | Investigate movement and distance rewards. |
# | 46–53 | Review behaviours and choose a recorded agent. |
# | 53–60 | Prepare Upload 2 and explain your comparisons. |
#
# Complete the baseline and **at least three distinct configurations**. The six suggested experiments offer different questions to investigate.

# %% [markdown]
# ## Setup
# Use a **CPU runtime** in Colab and run the next **three setup cells** in order. Setup downloads the public helpers automatically and checks their release hashes.
#
# Keep this runtime connected while working. Rerunning Setup 3 clears your experiment table, retained agents and current selection. Your saved submission files remain available.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Setup 1: prepare the CPU and load libraries { display-mode: "form" }
from datetime import datetime, timezone
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import types
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
import torch
from IPython.display import display, HTML, FileLink

CHALLENGE_VERSION = "snake-v2"
RELEASE_REVISION = "5f2d202e703da1308a3472f23761362980a26159"
MANIFEST_SHA256 = "a338e99aedf8695d4dc8a1acb9fafc983aeca601aa5381f19d76cd19840c9c8e"
RELEASE_BASE = f"https://raw.githubusercontent.com/eurecom-ds/intro-ai-challenges/{RELEASE_REVISION}/snake/"
ROOT = Path(globals().get("SNAKE_OUTPUT_DIR", Path(tempfile.gettempdir()) / "intro-ai-snake" / "exports"))
ROOT.mkdir(parents=True, exist_ok=True)
CACHE_ROOT = Path(tempfile.gettempdir()) / "intro-ai-snake" / "releases" / MANIFEST_SHA256
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
helper_override = globals().get("INSTRUCTOR_HELPER_DIR", os.environ.get("INSTRUCTOR_HELPER_DIR"))
HELPER_SOURCE = Path(helper_override).expanduser().resolve() if helper_override else None
if HELPER_SOURCE is None:
    for candidate in (Path.cwd(), Path.cwd() / "snake"):
        if (candidate / "data" / "manifest.json").is_file():
            HELPER_SOURCE = candidate.resolve()
            break

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "text.parse_math": False})
print("CPU ready. Run Setup 2 to load Snake.")


def checked_file(relative, expected_hash, expected_size=None):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("The helper release contains an invalid path or hash.")
    destination = CACHE_ROOT / path
    contents = destination.read_bytes() if destination.is_file() else None
    valid = (contents is not None and hashlib.sha256(contents).hexdigest() == expected_hash
             and (expected_size is None or len(contents) == expected_size))
    if valid:
        return destination
    limit = 100_001 if expected_size is None else expected_size + 1
    if HELPER_SOURCE is not None:
        source = HELPER_SOURCE / path
        if not source.is_file():
            raise ValueError(f"The local helper bundle is missing {relative}. Restore the complete approved bundle.")
        contents = source.read_bytes()
    else:
        try:
            with urllib.request.urlopen(RELEASE_BASE + relative, timeout=60) as response:
                contents = response.read(limit)
        except (OSError, urllib.error.URLError) as error:
            raise RuntimeError("Could not download " + relative + ". Check your connection and the challenge release, then rerun setup.") from error
    if expected_size is not None and len(contents) != expected_size:
        raise ValueError("Unexpected file size for " + relative + ". Restore the approved release file.")
    if hashlib.sha256(contents).hexdigest() != expected_hash:
        raise ValueError("The release hash differs for " + relative + ". Restore the approved helper bundle.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_helper_package(paths):
    name = "intro_ai_snake_" + MANIFEST_SHA256[:12]
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            del sys.modules[key]
    package = types.ModuleType(name)
    package.__path__ = [str(CACHE_ROOT)]
    sys.modules[name] = package
    loaded = {}
    for helper in ("protocol", "environment", "evaluator", "model"):
        module_name = name + "." + helper
        spec = importlib.util.spec_from_file_location(module_name, paths[helper + ".py"])
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        setattr(package, helper, module)
        spec.loader.exec_module(module)
        loaded[helper] = module
    return loaded


# %% cellView="form" jupyter={"source_hidden": true}
# @title Setup 2: verify and load the game, policy and evaluator { display-mode: "form" }
manifest_path = checked_file("data/manifest.json", MANIFEST_SHA256)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
helper_names = {"protocol.py", "environment.py", "evaluator.py", "model.py",
                "torch_environment.py", "numba_environment.py"}
if (manifest.get("challenge_version") != CHALLENGE_VERSION
        or set(manifest.get("files", {})) != helper_names):
    raise ValueError("The public files use another challenge format. Restore the approved release.")
paths = {}
for name in sorted(helper_names):
    entry = manifest["files"][name]
    if entry.get("path") != name or type(entry.get("size_bytes")) is not int or not 0 < entry["size_bytes"] <= 1_000_000:
        raise ValueError("The helper manifest contains an invalid file entry.")
    paths[name] = checked_file(name, entry["sha256"], entry["size_bytes"])
helpers = load_helper_package(paths)
snake_protocol = helpers["protocol"]
snake_game = helpers["environment"]
snake_evaluation = helpers["evaluator"]
snake_model = helpers["model"]
protocol_fingerprint = snake_protocol.protocol_fingerprint()
public_seed_pairs = snake_protocol.public_seed_pairs()
if (manifest.get("protocol_fingerprint") != protocol_fingerprint
        or manifest.get("public_seed_pairs") != [list(pair) for pair in public_seed_pairs]):
    raise ValueError("The helper code and public evaluation protocol differ. Restore one complete release.")
print(f"Snake ready: {snake_protocol.BOARD_SIZE} × {snake_protocol.BOARD_SIZE} board, "
      f"{snake_protocol.MAX_STEPS} moves per game, {snake_protocol.PUBLIC_EPISODES} public evaluation games.")


# %% cellView="form" jupyter={"source_hidden": true}
# @title Setup 3: prepare recorded agents, replays and uploads { display-mode: "form" }
experiments = []
last_result = None
configuration_selector = None
finalized_submission = None
upload_identity = None
upload_address = None
upload_access_code = None
HTML_FONT = "font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"


def show_notice(title, message, *, tone="info", items=()):
    accent, background = {"info": ("#2563eb", "#eff6ff"), "success": ("#15803d", "#f0fdf4"),
                          "attention": ("#b45309", "#fffbeb")}[tone]
    listing = ("<ul>" + "".join("<li>" + html.escape(str(item)) + "</li>" for item in items) + "</ul>") if items else ""
    display(HTML('<div style="' + HTML_FONT + 'margin:12px 0;padding:15px 18px;border:1px solid ' + accent +
                 ';border-left:5px solid ' + accent + ';border-radius:10px;background:' + background +
                 ';color:#172c4d;line-height:1.6;"><strong>' + html.escape(str(title)) +
                 '</strong><div>' + html.escape(str(message)) + "</div>" + listing + "</div>"))


def show_metrics(cards, *, title, detail=""):
    panels = []
    for label, value, note in cards:
        panels.append('<div style="flex:1 1 150px;padding:12px 16px;background:white;border-radius:8px;">'
                      '<div style="font-size:12px;color:#52647c;">' + html.escape(str(label)) +
                      '</div><div style="font-size:26px;font-weight:700;margin:4px 0;">' + html.escape(str(value)) +
                      '</div><div style="font-size:12px;color:#52647c;">' + html.escape(str(note)) + "</div></div>")
    display(HTML('<div style="' + HTML_FONT + 'margin:14px 0;padding:16px;background:#eff4fa;border-radius:10px;color:#172c4d;">'
                 "<strong>" + html.escape(str(title)) + '</strong><div style="font-size:12px;margin:5px 0 12px;">' +
                 html.escape(str(detail)) + '</div><div style="display:flex;flex-wrap:wrap;gap:10px;">' +
                 "".join(panels) + "</div></div>"))


def show_table(headers, rows):
    heading = "".join('<th style="text-align:left;padding:9px;background:#eff4fa;">' + html.escape(str(value)) + "</th>" for value in headers)
    body = "".join("<tr>" + "".join('<td style="padding:9px;border-bottom:1px solid #e2e8f0;">' +
                                   html.escape(str(value)) + "</td>" for value in row) + "</tr>" for row in rows)
    display(HTML('<div style="' + HTML_FONT + 'overflow-x:auto;margin:14px 0;color:#172c4d;">'
                 '<table style="border-collapse:collapse;min-width:960px;width:100%;font-size:12px;">'
                 "<thead><tr>" + heading + "</tr></thead><tbody>" + body + "</tbody></table></div>"))


def food_time(value):
    return "—" if value is None else f"{value:.1f}"


def configuration_key(config):
    return json.dumps(snake_protocol.settings(**config), sort_keys=True, separators=(",", ":"), allow_nan=False)


def describe_settings(config):
    return (f"{config['episodes']} games · food={config['food_reward']:g} · collision={config['collision_reward']:g} · "
            f"move={config['move_reward']:g} · distance={config['distance_reward']:g} · "
            f"γ={config['gamma']:g} · lr={config['learning_rate']:g}")


def recorded_run(key):
    return next((record for record in experiments if configuration_key(record["config"]) == key), None)


def best_record():
    return min(experiments, key=lambda record: snake_evaluation.ranking_key(record["validation"])) if experiments else None


def refresh_configuration_selector(preferred=None):
    if configuration_selector is None:
        return
    choice = configuration_selector.value if preferred is None else preferred
    options = [(f"Run {index} · {record['validation']['mean_apples']:.3f} apples/game · "
                f"food move {food_time(record['validation']['mean_food_step'])} · " + describe_settings(record["config"]),
                configuration_key(record["config"])) for index, record in enumerate(experiments, 1)]
    configuration_selector.options = options
    configuration_selector.disabled = not options
    if options:
        keys = [key for _, key in options]
        configuration_selector.value = choice if choice in keys else configuration_key(best_record()["config"])


def show_configuration_selector():
    global configuration_selector
    previous = configuration_selector.value if configuration_selector is not None else None
    if configuration_selector is not None:
        configuration_selector.disabled = True
        configuration_selector.close()
    if not experiments:
        configuration_selector = None
        show_notice("Run an experiment first", "The dropdown lists completed agents retained in this runtime.", tone="attention")
        return
    import ipywidgets as widgets
    try:
        from google.colab import output
    except ImportError:
        pass
    else:
        output.enable_custom_widget_manager()
    configuration_selector = widgets.Dropdown(description="Agent:", layout=widgets.Layout(width="100%"),
                                              style={"description_width": "initial"})
    refresh_configuration_selector(preferred=previous)
    display(configuration_selector)


def selected_record():
    if configuration_selector is None or configuration_selector.disabled:
        return None
    return recorded_run(configuration_selector.value)


def experiment_table():
    rows = []
    for index, record in enumerate(experiments, 1):
        config, score = record["config"], record["validation"]
        rows.append((index, config["episodes"], config["food_reward"], config["collision_reward"],
                     config["move_reward"], config["distance_reward"], config["gamma"], config["learning_rate"],
                     f"{score['mean_apples']:.3f}", food_time(score["mean_food_step"]),
                     record["checkpoint"]["training_steps"], f"{record['checkpoint']['training_seconds']:.1f}"))
    show_table(("Run", "Games", "Food", "Collision", "Move", "Distance", "γ", "Learning rate",
                "Apples/game ↑", "Food move ↓", "Training moves", "Seconds"), rows)


def draw_training(record):
    history = record["checkpoint"]["history"]
    training = [row for row in history if "training_return" in row]
    validation = [row for row in history if "validation" in row]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), layout="constrained")
    x = [row["episodes"] for row in training]
    axes[0, 0].plot(x, [row["training_return"] for row in training], color="#2563eb")
    axes[0, 0].set(title="Training return", ylabel="Raw reward per game")
    for key, label in (("food", "Food"), ("collision", "Collision"), ("move", "Each move"), ("distance", "Distance")):
        axes[0, 1].plot(x, [row["reward_components"][key] for row in training], label=label)
    axes[0, 1].set(title="Reward contributions", ylabel="Raw reward per game")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(x, [row["training_apples"] for row in training], color="#15803d")
    axes[1, 0].set(title="Training food collection", ylabel="Apples per game")
    axes[1, 1].plot([row["episodes"] for row in validation],
                    [row["validation"]["mean_apples"] for row in validation], "o-", color="#ea580c")
    axes[1, 1].set(title="Public evaluation", ylabel="Apples per game")
    for axis in axes.flat:
        axis.set_xlabel("Completed training games")
        axis.grid(alpha=0.15)
    plt.show()
    plt.close(fig)


def render_replay(replay, *, title):
    data = json.dumps(replay, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:14px system-ui,sans-serif;color:#172c4d;margin:12px}canvas{display:block;width:min(100%,360px);height:auto;margin:12px auto;background:#eff4fa;border-radius:8px}button{background:white;border:1px solid #94a3b8;border-radius:6px;padding:8px 14px;margin:0 5px 8px 0;color:#172c4d}button:focus-visible{outline:3px solid #2563eb}#status{margin:8px 0;min-height:36px}main{max-width:400px;margin:auto}</style>
<main><strong id="title"></strong><canvas width="360" height="360" aria-label="Snake game board"></canvas>
<button id="play">Play</button><button id="restart">Restart</button><span id="position"></span><div id="status" role="status"></div></main>
<script>
const replay=REPLAY_DATA,frames=replay.frames,canvas=document.querySelector('canvas'),ctx=canvas.getContext('2d');
document.getElementById('title').textContent=REPLAY_TITLE;
let position=0,timer=null;
function stop(){if(timer!==null){clearInterval(timer);timer=null;}document.getElementById('play').textContent='Play';}
function draw(){const f=frames[position],cell=canvas.width/replay.board_size;
ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#eff4fa';ctx.fillRect(0,0,canvas.width,canvas.height);
ctx.strokeStyle='#dce5ef';for(let i=0;i<=replay.board_size;i++){ctx.beginPath();ctx.moveTo(i*cell,0);ctx.lineTo(i*cell,canvas.height);ctx.stroke();ctx.beginPath();ctx.moveTo(0,i*cell);ctx.lineTo(canvas.width,i*cell);ctx.stroke();}
f.body.forEach((id,index)=>{ctx.fillStyle=index===f.body.length-1?'#2563eb':'#0d9488';ctx.fillRect((id%replay.board_size)*cell+3,Math.floor(id/replay.board_size)*cell+3,cell-6,cell-6);});
if(f.food!==null){ctx.fillStyle='#ea580c';ctx.beginPath();ctx.arc((f.food%replay.board_size+.5)*cell,(Math.floor(f.food/replay.board_size)+.5)*cell,cell*.22,0,Math.PI*2);ctx.fill();}
document.getElementById('position').textContent='Move '+f.step+' / '+replay.max_steps;
document.getElementById('status').textContent='Food collected: '+f.apples+(f.reason?' · '+f.reason.replaceAll('_',' '):' · game in progress');
}
document.getElementById('play').onclick=()=>{if(timer!==null){stop();return;}if(position===frames.length-1){position=0;draw();}document.getElementById('play').textContent='Pause';timer=setInterval(()=>{if(position<frames.length-1){position++;draw();}else{stop();}},100);};
document.getElementById('restart').onclick=()=>{stop();position=0;draw();};draw();
</script></html>"""
    document = document.replace("REPLAY_DATA", data).replace("REPLAY_TITLE", json.dumps(title).replace("<", "\\u003c"))
    display(HTML('<iframe title="' + html.escape(title, quote=True) + '" sandbox="allow-scripts" style="width:100%;'
                 'max-width:440px;height:530px;border:1px solid #dce5ef;border-radius:10px;" srcdoc="' +
                 html.escape(document, quote=True) + '"></iframe>'))


def preview_agent(preview_seed=7):
    if isinstance(preview_seed, bool) or not isinstance(preview_seed, int) or not 0 <= preview_seed < 2 ** 32:
        show_notice("Choose a preview seed", "Use a whole number from 0 to 2**32 − 1.", tone="attention")
        return
    record = selected_record() or last_result
    if record is None:
        show_notice("Train an agent first", "A preview uses your selected retained agent or your most recent completed run.", tone="attention")
        return
    result = snake_evaluation.evaluate(record["policy"], [(preview_seed, preview_seed ^ 0x51A7)], keep_replay=True)
    render_replay(result["replay"], title="Preview game: selected trained agent")
    print("Preview randomness is separate from the fixed public evaluation. The recorded score and saved upload stay fixed.")


def watch_untrained():
    policy = snake_model.initial_policy()
    result = snake_evaluation.evaluate(policy, [snake_protocol.DEMO_SEED_PAIR], keep_replay=True)
    render_replay(result["replay"], title="Untrained agent: sampled actions")


def run_experiment(episodes=8192, learning_rate=0.001, gamma=0.99, food_reward=10,
                   collision_reward=-10, move_reward=-0.01, distance_reward=0):
    global last_result
    try:
        config = snake_protocol.settings(episodes=episodes, learning_rate=learning_rate, gamma=gamma,
                                         food_reward=food_reward, collision_reward=collision_reward,
                                         move_reward=move_reward, distance_reward=distance_reward)
        key = configuration_key(config)
        existing = next((index for index, record in enumerate(experiments) if configuration_key(record["config"]) == key), None)
        if existing is None and len(experiments) >= 100:
            raise ValueError("The table contains 100 configurations. Compare retained agents or rerun Setup 3.")
        print("Training", describe_settings(config))
        progress_display = display(HTML("Preparing the first batch of complete games…"), display_id=True)
        last_refresh = 0.0

        def update(row):
            nonlocal last_refresh
            now = time.monotonic()
            if row["episodes"] != config["episodes"] and now - last_refresh < 0.25:
                return
            last_refresh = now
            message = (f"Completed {row['episodes']} / {config['episodes']} games · "
                       f"{row['steps']:,} training moves · {row['seconds']:.1f} seconds")
            if progress_display is not None:
                progress_display.update(HTML("<p>" + html.escape(message) + "</p>"))

        checkpoint = snake_model.train_policy(config, progress=update)
        score = checkpoint["validation"]
        if score["episodes"] != 30 or not np.isfinite(score["mean_apples"]):
            raise ValueError("The public evaluation is incomplete. Rerun this experiment.")
        policy = checkpoint["policy"]
        snake_evaluation.validate_policy(policy)
        record = {"config": config, "checkpoint": checkpoint, "policy": policy, "validation": score}
    except KeyboardInterrupt:
        show_notice("Training interrupted", "Earlier completed agents and your current selection are retained.", tone="attention")
        return None
    except (ValueError, TypeError, RuntimeError, FloatingPointError) as error:
        show_notice("Experiment not recorded", str(error), tone="attention")
        return None
    if existing is None:
        experiments.append(record)
    else:
        experiments[existing] = record
    last_result = record
    refresh_configuration_selector()
    show_metrics([("Apples/game ↑", f"{score['mean_apples']:.3f}", "30 fixed public games"),
                  ("Food move ↓", food_time(score["mean_food_step"]), "Tie-break when food totals match"),
                  ("Training moves", f"{checkpoint['training_steps']:,}", f"{config['episodes']} complete games")],
                 title="Final agent retained", detail=f"{len(experiments)} distinct configurations · {checkpoint['training_seconds']:.1f} seconds")
    draw_training(record)
    preview_agent()
    experiment_table()
    if finalized_submission is not None:
        print("Your saved submission stays frozen. Run an upload cell again to replace it.")
    return record


def completion_issues(selected=None):
    issues = []
    if recorded_run(configuration_key(snake_protocol.BASELINE)) is None:
        issues.append("Train the supplied 256-game starting agent.")
    if len(experiments) < 3:
        issues.append("Record at least three distinct configurations. Identical reruns count once.")
    if selected is not None and not any(selected is record for record in experiments):
        issues.append("Choose an agent retained in the experiment table.")
    return issues


def student_identity(student_name, student_surname):
    identity = {}
    for field, value in (("student_name", student_name), ("student_surname", student_surname)):
        label = "name" if field == "student_name" else "surname"
        if not isinstance(value, str):
            raise ValueError(f"Enter your {label} as text.")
        if any(unicodedata.category(character).startswith("C") or unicodedata.category(character) in ("Zl", "Zp") for character in value):
            raise ValueError(f"Enter your {label} without control characters.")
        value = " ".join(unicodedata.normalize("NFC", value).split())
        if not 1 <= len(value) <= 100:
            raise ValueError(f"Enter your {label} using 1 to 100 characters.")
        allowed = all(unicodedata.category(character)[0] in "LM" or unicodedata.category(character) == "Pd"
                      or character in " .'’‘" for character in value)
        if not allowed or value[0] in "=+-@" or not any(unicodedata.category(character).startswith("L") for character in value):
            raise ValueError(f"Enter your {label} using letters, spaces, apostrophes, hyphens or periods.")
        identity[field] = value
    return identity


def submission_filename(identity):
    readable = unicodedata.normalize("NFKD", " ".join(identity.values()).casefold()).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", readable).strip("-")[:60].strip("-") or "student"
    folded = {field: unicodedata.normalize("NFC", value.casefold()) for field, value in identity.items()}
    digest = hashlib.sha256(json.dumps(folded, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()[:12]
    return f"snake_{slug}_{digest}.json"


def make_fingerprint(payload, created_at=None):
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    body = {"created_at": created_at, "submission": {key: value for key, value in payload.items() if key != "fingerprint"}}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return {"created_at": created_at, "algorithm": "sha256", "value": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def finalize_submission(record, identity, *, phase="final"):
    global finalized_submission
    if record is None:
        show_notice("Choose a trained agent", "Train the starting agent or select a recorded configuration.", tone="attention")
        return False
    if phase == "baseline" and configuration_key(record["config"]) != configuration_key(snake_protocol.BASELINE):
        show_notice("Use the starting agent", "Upload 1 uses the supplied 256-game baseline.", tone="attention")
        return False
    if phase == "final":
        issues = completion_issues(record)
        if issues:
            show_notice("Before uploading", "Complete the remaining steps, then run Upload 2 again.", tone="attention", items=issues)
            return False
    score = record["validation"]
    payload = {"challenge_version": CHALLENGE_VERSION, "protocol_fingerprint": protocol_fingerprint,
               **identity, "phase": phase, "selected": record["config"],
               "validation": {field: int(score[field]) for field in ("episodes", "total_apples", "sum_food_steps")},
               "policy": record["policy"]}
    if finalized_submission is not None:
        previous_destination, previous_contents = finalized_submission
        previous = json.loads(previous_contents)
        if {key: value for key, value in previous.items() if key != "fingerprint"} == payload:
            previous_destination.write_bytes(previous_contents)
            show_notice("Saved submission reused", "The same agent and identity retain their timestamp and fingerprint for this retry.", tone="success")
            return True
    payload["fingerprint"] = make_fingerprint(payload)
    contents = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(contents) > snake_protocol.MAX_JSON_BYTES:
        raise ValueError("The export exceeds the challenge limit. Reload the approved helpers and rerun the experiment.")
    destination = ROOT / submission_filename(identity)
    destination.write_bytes(contents)
    finalized_submission = (destination, contents)
    show_notice("Submission saved", destination.name + " contains the selected final actor and its settings.", tone="success")
    return True


def upload_submission(api_url, token):
    if finalized_submission is None:
        show_notice("Prepare a submission first", "Run Upload 1 or Upload 2 to save its JSON.", tone="attention")
        return None
    destination, contents = finalized_submission

    def saved_copy():
        show_notice("Your file is saved", "Download the JSON below for manual upload or retry later.")
        display(FileLink(str(destination)))

    try:
        address = urllib.parse.urlsplit(api_url)
        if address.scheme not in {"https", "http"} or not address.hostname:
            raise ValueError("Set submission_url to the instructor's challenge submission endpoint.")
        if not isinstance(token, str) or not token.strip():
            raise ValueError("Enter the instructor's current access code in Upload 1.")
        request = urllib.request.Request(api_url, data=contents, method="POST",
                                         headers={"Content-Type": "application/json", "Accept": "application/json",
                                                  "Authorization": "Bearer " + token.strip()})

        class DirectUpload(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        with urllib.request.build_opener(DirectUpload()).open(request, timeout=30) as response:
            if response.status not in {200, 201}:
                raise ValueError("The server did not confirm the upload.")
            body = response.read(65_537)
        if len(body) > 65_536:
            raise ValueError("Unexpected server response. Check the dashboard before uploading again.")
        result = json.loads(body.decode("utf-8"))
        best = result.get("best") if isinstance(result, dict) else None
        for item in (result, best):
            if not isinstance(item, dict):
                raise ValueError("No readable server metrics were received.")
            for field in ("mean_apples",):
                value = item.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                    raise ValueError("No readable server metrics were received.")
            value = item.get("mean_food_step")
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 1):
                raise ValueError("No readable collection time was received.")
    except urllib.error.HTTPError as error:
        if error.code in {401, 403}:
            message = "Enter the instructor's current access code in Upload 1 and try again."
        elif error.code in {301, 302, 303, 307, 308, 404}:
            message = "Check the submission endpoint with your instructor."
        elif error.code >= 500:
            message = "The scoring service is unavailable. Keep your JSON and retry later."
        else:
            message = f"The upload was not accepted (HTTP {error.code})."
            try:
                detail = json.loads(error.read(65_536).decode("utf-8")).get("detail")
                if isinstance(detail, str):
                    message += " " + detail[:500]
                elif isinstance(detail, dict):
                    message += " " + str(detail.get("message", ""))[:500]
                    for issue in detail.get("issues", [])[:8]:
                        if isinstance(issue, str):
                            print(issue[:300])
            except (OSError, ValueError, TypeError, UnicodeError, AttributeError):
                pass
        error.close()
        show_notice("Upload needs attention", message, tone="attention")
        saved_copy()
        return None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, TypeError, KeyError, UnicodeError) as error:
        show_notice("Upload needs attention", str(error) if isinstance(error, ValueError) else
                    "The service could not be reached or returned an unreadable result. Keep your saved JSON.", tone="attention")
        saved_copy()
        return None
    show_metrics([("This upload", f"{result['mean_apples']:.3f} apples/game", "Food move " + food_time(result["mean_food_step"])),
                  ("Your best", f"{best['mean_apples']:.3f} apples/game", "Food move " + food_time(best["mean_food_step"]))],
                 title="Already uploaded" if result.get("duplicate") else "Upload confirmed",
                 detail="Private games scored by the server. More food ranks first; equal food totals favour earlier collection.")
    dashboard = result.get("dashboard_url")
    if isinstance(dashboard, str):
        url = urllib.parse.urljoin(api_url, dashboard)
        target = urllib.parse.urlsplit(url)
        if (target.scheme, target.netloc) == (address.scheme, address.netloc):
            display(HTML('<a href="' + html.escape(url, quote=True) + '" target="_blank" rel="noopener noreferrer">Open your challenge dashboard →</a>'))
    return result


def upload_baseline(student_name, student_surname, api_url, token):
    global upload_identity, upload_address, upload_access_code
    try:
        identity = student_identity(student_name, student_surname)
        record = recorded_run(configuration_key(snake_protocol.BASELINE))
        if not finalize_submission(record, identity, phase="baseline"):
            return None
        upload_identity, upload_address, upload_access_code = identity, api_url, token
        return upload_submission(api_url, token)
    except KeyboardInterrupt:
        show_notice("Upload paused", "Your JSON is saved. Run Upload 1 again when ready.", tone="attention")
    except (ValueError, TypeError, OSError) as error:
        show_notice("Before uploading", str(error), tone="attention")
    return None


def upload_selected():
    if upload_identity is None:
        show_notice("Complete Upload 1 first", "Enter your name, surname and instructor's code in Upload 1.", tone="attention")
        return None
    try:
        if finalize_submission(selected_record(), upload_identity, phase="final"):
            return upload_submission(upload_address, upload_access_code)
    except (ValueError, TypeError, OSError) as error:
        show_notice("Could not prepare the upload", str(error), tone="attention")
    return None


def export_submission():
    if finalized_submission is None:
        show_notice("Prepare an upload first", "Run Upload 1 or Upload 2 to save the submission JSON.", tone="attention")
        return
    destination, contents = finalized_submission
    destination.write_bytes(contents)
    try:
        from google.colab import files
    except ImportError:
        display(FileLink(str(destination)))
    else:
        files.download(str(destination))


show_notice("Ready to play", "Inspect the game, train the starting agent and prepare Upload 1 before changing settings.", tone="success")

# %% [markdown]
# ## 1. Inspect the game
# ![A 15 × 15 board, three actions and a growing snake.](https://raw.githubusercontent.com/eurecom-ds/intro-ai-challenges/5f2d202e703da1308a3472f23761362980a26159/snake/assets/game.svg)
#
# The snake starts with three cells. It moves straight, turns left or turns right. Eating an orange food item grows its body. Hitting a wall or its body ends the game. Each game lasts at most **1,500 moves**.
#
# The agent receives a numeric board description: the ordered body, head, food, heading, remaining moves, food offsets and immediate collision indicators. Body order tells it which cells the tail will leave next. Its three actions remain available, including actions that cause a collision.
#
# Watch an untrained policy. Its action probabilities are initially close to one third each. What would you expect it to learn first?

# %% cellView="form" jupyter={"source_hidden": true}
# @title Watch an untrained agent { display-mode: "form" }
watch_untrained()

# %% [markdown]
# ## 2. Understand policy gradients and train the baseline
# ![Train the starting agent. Record its behaviour and public evaluation.](https://raw.githubusercontent.com/eurecom-ds/intro-ai-challenges/5f2d202e703da1308a3472f23761362980a26159/snake/assets/baseline.svg)
#
# A **policy** assigns probabilities to actions from the current observation. During training, the agent samples an action, receives a reward and continues until the game ends.
#
# **REINFORCE** uses the rewards that follow an action to update its probability. The **discount factor** γ controls how strongly later rewards contribute. A learned **value baseline** estimates the return expected from an observation. The difference between observed return and this estimate tells the update whether the outcome was better or worse than expected.
#
# Training collects batches of **32 complete games**, then updates the policy and baseline. Each experiment starts from the same initialization and random seeds. Training retains its final policy.
#
# The starting configuration trains for **256 games**, with food reward **10**, collision reward **−10**, move reward **−0.01** and distance reward **0**.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Train the 256-game starting agent { display-mode: "form" }
run_experiment(**{key: value for key, value in snake_protocol.BASELINE.items() if key != "seed"})

# %% [markdown]
# ### How evaluation works
# The notebook evaluates each agent on **30 fixed public games**, using sampled actions. The server evaluates its uploaded weights on **100 separate private games** from the same game distribution.
#
# Rank agents by total food collected. When totals match, prefer the lower sum of food-collection moves. An apple collected on move 10 contributes 10 to this sum; one collected on move 80 contributes 80. The table displays **apples/game** and **average food-collection move**. A game with no food has no collection-time value.
#
# For example, four apples outrank three at any speed. Between two agents collecting four apples, the agent with the lower total collection time wins.
#
# Training rewards guide learning. Food and collection times provide the common evaluation objective.

# %% [markdown]
# ![Upload 1. Save the baseline policy, then send it for server evaluation.](https://raw.githubusercontent.com/eurecom-ds/intro-ai-challenges/5f2d202e703da1308a3472f23761362980a26159/snake/assets/upload-baseline.svg)
# Enter your name, surname and the current code from your instructor. Keep the supplied address unless your instructor gives you another one. Upload 2 reuses these fields. Your name and surname appear on the public course dashboard.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Upload 1: save and upload the starting agent { display-mode: "form" }
# @markdown Enter your name, surname, submission URL and the current code from your instructor. Upload 2 reuses these fields.
name = "" # @param {type:"string"}
surname = "" # @param {type:"string"}
submission_url = "https://eurecom-probai-intro-ai-leaderboard.hf.space/api/challenges/snake/submissions" # @param {type:"string"}
access_code = "" # @param {type:"string"}
upload_baseline(name, surname, submission_url, access_code)

# %% [markdown]
# ## 3. Investigate the rewards
# ![Predict, change one setting, train, then inspect behaviour and food collection.](https://raw.githubusercontent.com/eurecom-ds/intro-ai-challenges/5f2d202e703da1308a3472f23761362980a26159/snake/assets/investigations.svg)
#
# The reward after each attempted move is
#
# $$
# r_t=w_{\mathrm{food}}I_{\mathrm{food}}+
# w_{\mathrm{collision}}I_{\mathrm{collision}}+
# w_{\mathrm{move}}+
# w_{\mathrm{distance}}\Delta d_t.
# $$
#
# The two indicators equal 1 when food is eaten or a collision occurs. The move reward applies on every attempt, including a collision. Distance progress is the decrease in **Manhattan distance**, the number of horizontal and vertical moves to the current food. A move closer gives +1; a move farther gives −1. On an eating move, progress uses the food's old position before replacement food appears. Collision progress is zero.
#
# Every reward control accepts positive, zero and negative values. Before changing a value, predict the behaviour it could encourage. Train, watch a preview and compare public food collection.
#
# | Suggested run | Training games | Food | Collision | Move | Distance | Question |
# |---|---:|---:|---:|---:|---:|---|
# | Starting agent | 256 | 10 | −10 | −0.01 | 0 | What does short training achieve? |
# | Reference | 8192 | 10 | −10 | −0.01 | 0 | What changes with more experience? |
# | Food | 8192 | 1 | −10 | −0.01 | 0 | How does a smaller food reward change behaviour? |
# | Collision | 8192 | 10 | 0 | −0.01 | 0 | What happens when collisions have no direct penalty? |
# | Movement | 8192 | 10 | −10 | +0.1 | 0 | What behaviour earns movement rewards? |
# | Distance | 8192 | 10 | −10 | −0.01 | 1 | Does progress feedback help collect food? |
#
# Use learning rate **0.001** and γ **0.99** for these comparisons. Change one setting at a time from the reference configuration.
#
# Complete **at least three distinct configurations, including the starting agent**. The remaining suggested runs and learning-rate or discount-factor changes are further investigations.
#
# The charts show raw training return, its four contributions, training food collection and public evaluation. Compare configurations with different reward values through food and collection time. Training return has the units and preferences of the chosen reward function.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Choose rewards and settings, then train an experiment { display-mode: "form" }
episodes = 8192 # @param [256, 4096, 8192, 16384] {type:"raw"}
food_reward = 10 # @param {type:"number"}
collision_reward = -10 # @param {type:"number"}
move_reward = -0.01 # @param {type:"number"}
distance_reward = 0 # @param {type:"number"}
learning_rate = 0.001 # @param [0.0001, 0.0003, 0.001, 0.003, 0.01] {type:"raw"}
gamma = 0.99 # @param [0, 0.5, 0.9, 0.95, 0.99, 1] {type:"raw"}
run_experiment(episodes, learning_rate, gamma, food_reward, collision_reward, move_reward, distance_reward)

# %% [markdown]
# ## 4. Inspect another game
# The preview uses the selected agent once you open the selector below. Before that, it uses the latest completed run. Change the preview seed to inspect another game. This leaves the recorded public evaluation and saved submission fixed.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Watch the current agent with another preview seed { display-mode: "form" }
preview_seed = 7 # @param {type:"integer"}
preview_agent(preview_seed)

# %% [markdown]
# ## 5. Select a recorded agent
# ![Compare food collected, collection speed and observed behaviour, then select a retained agent.](https://raw.githubusercontent.com/eurecom-ds/intro-ai-challenges/5f2d202e703da1308a3472f23761362980a26159/snake/assets/selection.svg)
#
# Review the public metrics and replays. The dropdown initially selects the agent with the most food, using collection speed to break food ties. You can choose any retained agent. New experiments refresh the options while preserving your choice.
#
# Each row represents one distinct configuration and its final trained actor. Identical reruns replace that row. Training games contain different numbers of moves, so use the recorded move counts when discussing experience.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Review experiments and select a recorded agent { display-mode: "form" }
experiment_table()
show_configuration_selector()

# %% [markdown]
# ## 6. Upload the selected policy
# ![Upload 2. Save the selected retained policy and evaluate it on private games.](https://raw.githubusercontent.com/eurecom-ds/intro-ai-challenges/5f2d202e703da1308a3472f23761362980a26159/snake/assets/upload-selected.svg)
#
# Complete the baseline and at least three distinct configurations, then choose an agent above. Upload 2 uses its retained final weights and performs no further training. Change the dropdown to upload another retained agent.
#
# Upload 2 reuses your details from Upload 1. To correct your name, surname, address or access code, edit the fields in Upload 1 and run it again. Your retained agents stay available for selection.
#
# The JSON contains numeric policy weights, reported settings and public metrics. The server runs the trusted game simulator and computes the ranked private metrics.
#
# Both uploads save a **snake_name_hash.json** file. If the service is unavailable, download it below and use the dashboard's manual upload form later. An unchanged retry keeps its timestamp and fingerprint. The server retains accepted score history.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Upload 2: save and upload the selected agent { display-mode: "form" }
upload_selected()

# %% [markdown]
# ### Keep a copy and discuss
# Prepare to explain which reward changes affected behaviour, where training return and food collection differed, and why you chose this agent. Use examples from its games.

# %% cellView="form" jupyter={"source_hidden": true}
# @title Download the saved submission JSON { display-mode: "form" }
export_submission()
