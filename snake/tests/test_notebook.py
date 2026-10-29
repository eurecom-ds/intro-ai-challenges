"""Execute the notebook setup and retained-policy upload flow locally."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebook.py"


def source_cells():
    pieces = re.split(r"(?m)^# %%([^\n]*)\n", SOURCE.read_text(encoding="utf-8"))
    return [(pieces[index], pieces[index + 1]) for index in range(1, len(pieces), 2)]


class DisplayHandle:
    def update(self, value):
        pass


class NotebookFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="snake-notebook-tests-")
        cls.directory = Path(cls.temporary.name)
        cls.bundle = cls.directory / "bundle"
        (cls.bundle / "data").mkdir(parents=True)
        for name in ("protocol.py", "environment.py", "evaluator.py", "model.py"):
            shutil.copyfile(ROOT / name, cls.bundle / name)
        sys.path.insert(0, str(ROOT))
        import protocol
        files = {}
        for path in cls.bundle.glob("*.py"):
            raw = path.read_bytes()
            files[path.name] = {"path": path.name, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        manifest = {"challenge_version": protocol.CHALLENGE_VERSION, "protocol_fingerprint": protocol.protocol_fingerprint(),
                    "files": files, "public_seed_pairs": [list(pair) for pair in protocol.public_seed_pairs()]}
        raw = json.dumps(manifest, sort_keys=True).encode()
        (cls.bundle / "data" / "manifest.json").write_bytes(raw)
        cls.manifest_hash = hashlib.sha256(raw).hexdigest()
        cls.setup = [source for metadata, source in source_cells() if "[markdown]" not in metadata][:3]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.case = Path(tempfile.mkdtemp(dir=self.directory))
        self.displays = []
        self.g = {"__name__": "__main__", "INSTRUCTOR_HELPER_DIR": str(self.bundle),
                  "SNAKE_OUTPUT_DIR": str(self.case / "exports"),
                  "SNAKE_API_URL": "http://127.0.0.1:7861/api/challenges/snake/submissions"}
        with patch("sys.stdout", new=io.StringIO()):
            exec(compile(self.setup[0], str(SOURCE), "exec"), self.g)
        self.g["MANIFEST_SHA256"] = self.manifest_hash
        self.g["CACHE_ROOT"] = self.case / "cache"
        self.g["display"] = lambda value, **kwargs: self.displays.append(value) or DisplayHandle()
        exec(compile(self.setup[1], str(SOURCE), "exec"), self.g)
        exec(compile(self.setup[2], str(SOURCE), "exec"), self.g)

    def silent_run(self, **kwargs):
        with patch("sys.stdout", new=io.StringIO()):
            return self.g["run_experiment"](**kwargs)

    def fast_trainer(self, config, progress=None):
        if progress:
            progress({"episodes": config["episodes"], "steps": self.g["snake_protocol"].MAX_STEPS, "seconds": 0.01})
        policy = self.g["snake_evaluation"].zero_policy()
        apples = 2 if config["food_reward"] != 10 else 1
        validation = {"episodes": 30, "total_apples": apples, "sum_food_steps": 10 * apples,
                      "mean_apples": apples / 30, "mean_food_step": 10.0,
                      "diagnostics": {}}
        return {"config": config, "policy": policy, "history": [], "training_steps": self.g["snake_protocol"].MAX_STEPS,
                "training_episodes": config["episodes"], "training_seconds": 0.01, "validation": validation}

    def mock_rendering(self):
        self.g["draw_training"] = lambda record: None
        self.g["preview_agent"] = lambda preview_seed=7: None

    def test_cells_compile_and_metadata(self):
        cells = source_cells()
        code = [source for metadata, source in cells if "[markdown]" not in metadata]
        self.assertEqual(len(self.setup), 3)
        for index, cell in enumerate(code):
            compile(cell, f"notebook-cell-{index}", "exec")
        self.assertTrue(all("# @title Setup " in cell for cell in code[:3]))
        source = SOURCE.read_text()
        self.assertIn("at least three distinct configurations", source.lower())
        self.assertNotIn("ROOT_FILL_", source)
        self.assertNotIn("accelerator: GPU", source)

    def test_local_bundle_and_scoped_refresh(self):
        self.assertEqual(self.g["snake_protocol"].OBSERVATION_SIZE, 685)
        first = self.g["snake_model"]
        self.assertTrue(first.__name__.startswith("intro_ai_snake_"))
        exec(self.setup[1], self.g)
        self.assertIsNot(self.g["snake_model"], first)
        self.assertEqual(self.g["protocol_fingerprint"], self.g["manifest"]["protocol_fingerprint"])
        for filename in ("protocol.py", "environment.py", "evaluator.py", "model.py"):
            self.assertEqual((self.g["CACHE_ROOT"] / filename).read_bytes(), (self.bundle / filename).read_bytes())

    def test_hash_failure_preserves_cached_bytes(self):
        path = self.g["CACHE_ROOT"] / "model.py"
        previous = path.read_bytes()
        with self.assertRaises(ValueError):
            self.g["checked_file"]("model.py", "0" * 64, len(previous))
        self.assertEqual(path.read_bytes(), previous)

    def test_real_baseline_retains_final_actor(self):
        self.mock_rendering()
        baseline = self.g["snake_protocol"].BASELINE
        record = self.silent_run(**{key: value for key, value in baseline.items() if key != "seed"})
        self.assertIsNotNone(record)
        self.assertEqual(record["config"], baseline)
        self.assertEqual(record["checkpoint"]["training_episodes"], 256)
        self.assertEqual(record["policy"], record["checkpoint"]["policy"])
        self.assertEqual(self.g["completion_issues"](), ["Record at least three distinct configurations. Identical reruns count once."])

    def test_record_selection_completion_and_interruptions(self):
        self.mock_rendering()
        self.g["snake_model"].train_policy = self.fast_trainer
        baseline = self.silent_run(episodes=256)
        self.assertIsNotNone(baseline)
        self.silent_run()
        self.assertEqual(len(self.g["experiments"]), 2)
        self.assertTrue(self.g["completion_issues"]())
        self.g["show_configuration_selector"]()
        selected_before = self.g["configuration_selector"].value
        third = self.silent_run(food_reward=1)
        self.assertEqual(len(self.g["experiments"]), 3)
        self.assertEqual(self.g["completion_issues"](third), [])
        self.assertEqual(self.g["configuration_selector"].value, selected_before)
        self.assertIs(self.g["best_record"](), third)
        self.silent_run(food_reward=1)
        self.assertEqual(len(self.g["experiments"]), 3)
        self.g["snake_model"].train_policy = lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt())
        selected_before = self.g["configuration_selector"].value
        self.assertIsNone(self.silent_run(food_reward=2))
        self.assertEqual(len(self.g["experiments"]), 3)
        self.assertEqual(self.g["configuration_selector"].value, selected_before)
        self.assertIsNone(self.silent_run(food_reward=float("nan")))
        self.assertEqual(len(self.g["experiments"]), 3)

    def test_identity_export_retry_and_server_validation(self):
        self.mock_rendering()
        self.g["snake_model"].train_policy = self.fast_trainer
        baseline = self.silent_run(episodes=256)
        self.silent_run()
        third = self.silent_run(food_reward=1)
        identity = self.g["student_identity"]("  Élodie  ", "O’Neill")
        self.assertEqual(identity, {"student_name": "Élodie", "student_surname": "O’Neill"})
        with self.assertRaises(ValueError):
            self.g["student_identity"]("=formula", "Person")
        self.assertTrue(self.g["finalize_submission"](third, identity, phase="final"))
        destination, contents = self.g["finalized_submission"]
        self.assertTrue(destination.name.startswith("snake_elodie-oneill_"))
        self.assertEqual(destination.read_bytes(), contents)
        self.assertTrue(self.g["finalize_submission"](third, identity, phase="final"))
        self.assertEqual(self.g["finalized_submission"][1], contents)
        payload = json.loads(contents)
        self.assertEqual(set(payload), {"challenge_version", "protocol_fingerprint", "student_name", "student_surname",
                                       "phase", "selected", "validation", "policy", "fingerprint"})
        self.assertEqual(payload["fingerprint"], self.g["make_fingerprint"](payload, payload["fingerprint"]["created_at"]))
        sys.path.insert(0, str(ROOT.parents[1] / "intro-ai-leaderboard"))
        from challenges.snake.challenge_core import validate_payload, submission_filename
        validated = validate_payload(payload, {"protocol_fingerprint": self.g["protocol_fingerprint"]})
        self.assertEqual(validated["selected"], third["config"])
        self.assertEqual(submission_filename(**identity), destination.name)
        self.assertFalse(self.g["finalize_submission"](third, identity, phase="baseline"))
        self.assertTrue(self.g["finalize_submission"](baseline, identity, phase="baseline"))

    def test_upload_failure_saves_identity_and_export(self):
        self.mock_rendering()
        self.g["snake_model"].train_policy = self.fast_trainer
        self.silent_run(episodes=256)
        opener = unittest.mock.Mock()
        opener.open.side_effect = urllib.error.URLError("offline")
        with patch("urllib.request.build_opener", return_value=opener):
            result = self.g["upload_baseline"]("Test", "Learner", token="class42")
        self.assertIsNone(result)
        self.assertEqual(self.g["upload_identity"]["student_name"], "Test")
        self.assertEqual(self.g["upload_access_code"], "class42")
        destination, raw = self.g["finalized_submission"]
        self.assertTrue(destination.is_file())
        self.assertNotIn(b"class42", raw)
        self.assertEqual(json.loads(raw)["phase"], "baseline")
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer class42")
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 30)
        self.g["upload_submission"]("http://example.com/upload", "class42")
        self.assertEqual(destination.read_bytes(), raw)

    def test_confirmed_upload_and_redirect_handler(self):
        self.mock_rendering()
        self.g["snake_model"].train_policy = self.fast_trainer
        self.silent_run(episodes=256)
        current = {"mean_apples": 2.0, "mean_food_step": 10.0}
        result = {**current, "best": current, "duplicate": True,
                  "dashboard_url": "/challenges/snake?student=learner-0123456789"}
        response = unittest.mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.read.return_value = json.dumps(result).encode()
        opener = unittest.mock.Mock()
        opener.open.return_value = response
        with patch("urllib.request.build_opener", return_value=opener) as build:
            actual = self.g["upload_baseline"]("Test", "Learner", token="class42")
        self.assertEqual(actual, result)
        handler = build.call_args.args[0]
        self.assertIsNone(handler.redirect_request(None, None, 302, "", {}, "https://example.com"))

    def test_replay_is_sandboxed_and_labelled(self):
        self.g["watch_untrained"]()
        markup = self.displays[-1].data
        self.assertIn('sandbox="allow-scripts"', markup)
        self.assertIn("Untrained agent: sampled actions", markup)
        self.assertIn("Restart", markup)
        self.assertIn("Food collected", markup)


if __name__ == "__main__":
    unittest.main()
