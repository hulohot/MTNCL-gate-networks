from mtncl_nn import MTNCLNetwork


def test_forward_returns_one_hot_vector():
    net = MTNCLNetwork(num_inputs=2, num_outputs=2, hidden_layers=[2])
    out = net.forward([1, 0])
    assert len(out) == 2
    assert sum(out) == 1.0
    assert all(v in (0.0, 1.0) for v in out)


def test_train_smoke_and_evaluate():
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[1, 0], [0, 1], [0, 1], [1, 0]]
    net = MTNCLNetwork(num_inputs=2, num_outputs=2, hidden_layers=[3])
    net.train(X, y, iterations=5)
    err, acc = net.evaluate(X, y)
    assert 0.0 <= err <= 1.0
    assert 0.0 <= acc <= 1.0


def test_save_and_load_roundtrip(tmp_path):
    net = MTNCLNetwork(num_inputs=2, num_outputs=2, hidden_layers=[2])
    model_path = tmp_path / "model.json"
    net.save(str(model_path))

    loaded = MTNCLNetwork.load(str(model_path))
    out = loaded.forward([1, 0])
    assert len(out) == 2
    assert sum(out) == 1.0
