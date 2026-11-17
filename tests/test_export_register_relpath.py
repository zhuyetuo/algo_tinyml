import os, sys, json
sys.path.insert(0, "service")
import export_train as et
def test_register_local_relative(tmp_path):
    lj = tmp_path / "edge_models.local.json"
    gen = tmp_path / "core" / "models" / "generated_x"
    et.register_local("x", str(gen), str(gen / "meta.json"), str(lj), kind="cnn")
    m = json.load(open(lj))["models"][0]
    assert m["gen"] == os.path.join("core", "models", "generated_x")
    et.register_local("y", "/elsewhere/g", "/elsewhere/g/meta.json", str(lj), kind="cnn")
    assert json.load(open(lj))["models"][-1]["gen"] == "/elsewhere/g"
