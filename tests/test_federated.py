# tests/test_federated.py
"""
Smoke test: 2-round federated simulation with mock data.
Verifies the full FL loop works before running on real data.
"""
import sys, json, os
sys.path.insert(0, ".")

def test_federated():
    print("\n" + "="*55)
    print("  CHECKPOINT 4 — Federated Learning Engine Test")
    print("="*55)

    from src.federated.server import run_federated_training

    # Run 2-round smoke test with mock data
    results = run_federated_training(
        tbx_root      = "data/raw/TBX11K_FAKE",  # forces mock
        shenzhen_root = "data/raw/Shenzhen_FAKE",
        num_rounds    = 2,
        strategy_name = "fedprox",
        local_epochs  = 1,       # 1 epoch to keep test fast
        batch_size    = 16,
        use_mock      = True,
        results_path  = "runs/test_results.json",
        verbose       = True,
    )

    # Verify structure
    assert "round_metrics" in results, "❌ No round_metrics in results"
    assert len(results["round_metrics"]) == 2, \
        f"❌ Expected 2 rounds, got {len(results['round_metrics'])}"

    for r in results["round_metrics"]:
        assert "avg_train_accuracy" in r, "❌ Missing accuracy in round metrics"
        assert 0.0 <= r["avg_train_accuracy"] <= 1.0, \
            f"❌ Accuracy out of range: {r['avg_train_accuracy']}"

    print("\n  Round summary:")
    for r in results["round_metrics"]:
        print(f"    Round {r['round']}: "
              f"avg_train_acc={r['avg_train_accuracy']:.4f} | "
              f"strategy={r['strategy']}")

    # Verify results file was saved
    assert os.path.isfile("runs/test_results.json"), \
        "❌ Results JSON not saved"
    print("\n  ✓ Results JSON saved correctly")

    print("\n✅  CHECKPOINT 4 PASSED — Federated engine is ready!\n")

if __name__ == "__main__":
    test_federated()