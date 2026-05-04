# tests/test_model.py
import torch
import sys
sys.path.insert(0, ".")

from src.model.resnet_tb import build_model
from src.model.model_utils import (
    get_model_weights, set_model_weights, print_model_summary
)

def test_model():
    print("\n" + "="*50)
    print("  CHECKPOINT 2 — Model Architecture Test")
    print("="*50)

    # 1. Build model
    model = build_model(num_classes=3, pretrained=True)
    print_model_summary(model)

    # 2. Verify freeze policy
    stats = model.get_param_stats()
    assert stats["frozen"] > 0,    "❌ No layers are frozen!"
    assert stats["trainable"] > 0, "❌ No layers are trainable!"
    print(f"  ✓ Freeze policy correct")

    # 3. Forward pass with dummy X-ray batch
    dummy_batch = torch.randn(4, 3, 224, 224)   # batch=4, RGB, 224×224
    output = model(dummy_batch)
    assert output.shape == (4, 3), \
        f"❌ Wrong output shape: {output.shape}, expected (4, 3)"
    print(f"  ✓ Forward pass OK — output shape: {output.shape}")

    # 4. Softmax → confidence percentages
    probs = torch.softmax(output, dim=1)
    assert probs.shape == (4, 3)
    assert abs(probs[0].sum().item() - 1.0) < 1e-5, "❌ Probs don't sum to 1"
    classes = ["Healthy", "Active TB", "Latent TB"]
    print(f"  ✓ Softmax OK — sample prediction:")
    for cls, prob in zip(classes, probs[0].tolist()):
        print(f"      {cls:<12}: {prob*100:.1f}%")

    # 5. Weight extraction / injection (used by Flower FL)
    weights = get_model_weights(model)
    print(f"\n  ✓ Extracted {len(weights)} trainable weight tensors")

    model2 = build_model(num_classes=3, pretrained=False)
    set_model_weights(model2, weights)
    weights2 = get_model_weights(model2)
    for w1, w2 in zip(weights, weights2):
        assert torch.allclose(w1, w2), "❌ Weight injection failed!"
    print(f"  ✓ Weight get/set round-trip OK (Flower FL compatible)")

    # 6. Trainable param count sanity check (~4.7M for layer3+4+fc)
    assert 9_000_000 < stats["trainable"] < 11_500_000, \
        f"❌ Unexpected trainable param count: {stats['trainable']:,}"
    print(f"  ✓ Param counts look correct")

    print("\n✅  CHECKPOINT 2 PASSED — Model is ready!\n")

if __name__ == "__main__":
    test_model()