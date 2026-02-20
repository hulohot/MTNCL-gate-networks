import json
import subprocess
import sys


def test_cli_train_and_infer(tmp_path):
    dataset = {
        "X": [[0, 0], [0, 1], [1, 0], [1, 1]],
        "y": [[1, 0], [0, 1], [0, 1], [1, 0]],
    }
    data_path = tmp_path / "data.json"
    model_path = tmp_path / "model.json"
    inputs_path = tmp_path / "inputs.json"
    output_path = tmp_path / "preds.json"

    data_path.write_text(json.dumps(dataset), encoding="utf-8")
    inputs_path.write_text(json.dumps(dataset["X"]), encoding="utf-8")

    train_cmd = [
        sys.executable,
        "-m",
        "mtncl_nn.cli",
        "train",
        "--data",
        str(data_path),
        "--output",
        str(model_path),
        "--num-inputs",
        "2",
        "--num-outputs",
        "2",
        "--hidden-layers",
        "2",
        "--iterations",
        "2",
    ]
    infer_cmd = [
        sys.executable,
        "-m",
        "mtncl_nn.cli",
        "infer",
        "--model",
        str(model_path),
        "--inputs",
        str(inputs_path),
        "--output",
        str(output_path),
    ]

    subprocess.run(train_cmd, check=True)
    subprocess.run(infer_cmd, check=True)

    preds = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(preds) == 4
    assert all(len(row) == 2 for row in preds)
