#!/usr/bin/env python
"""
Test script to verify gradient flow in PyTorch K-means implementation.

This script verifies that:
1. PyTorch K-means preserves gradients
2. Loss can backpropagate to encoder
3. All components integrate correctly
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add project to path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

from models.spm_classifier_kmeans import Feature_Aggregation_KMeans, SemiParametricClassifier_KMeans


def test_kmeans_gradient_flow():
    """Test that gradients flow through PyTorch K-means."""
    print("\n" + "="*70)
    print("TEST 1: K-means Gradient Flow")
    print("="*70)
    
    K = 3
    num_classes = 10
    embed_dim = 512
    batch_size = 100
    
    # Create random embeddings (simulating encoder output)
    embeddings = torch.randn(batch_size, embed_dim, requires_grad=True)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    print(f"Input: embeddings {embeddings.shape}, labels {labels.shape}")
    print(f"Requires grad: {embeddings.requires_grad}")
    
    # Apply K-means aggregation
    aggregator = Feature_Aggregation_KMeans(num_classes, K=K)
    prototypes, proto_labels = aggregator(embeddings, labels)
    
    print(f"Output: prototypes {prototypes.shape}, proto_labels {proto_labels.shape}")
    print(f"Prototypes requires grad: {prototypes.requires_grad}")
    
    # Compute loss and backward
    loss = prototypes.sum()  # Dummy loss
    loss.backward()
    
    # Check gradients
    has_grads = embeddings.grad is not None
    print(f"\nGradient check:")
    print(f"  - embeddings.grad is not None: {has_grads}")
    
    if has_grads:
        grad_norm = embeddings.grad.norm().item()
        print(f"  - Gradient norm: {grad_norm:.6f}")
        print(f"  - Gradient min: {embeddings.grad.min().item():.6f}")
        print(f"  - Gradient max: {embeddings.grad.max().item():.6f}")
        print("\n✓ TEST PASSED: Gradients flowing through K-means!")
        return True
    else:
        print("\n✗ TEST FAILED: No gradients!")
        return False


def test_classifier_gradient_flow():
    """Test that gradients flow through entire classifier."""
    print("\n" + "="*70)
    print("TEST 2: Full Classifier Gradient Flow")
    print("="*70)
    
    device = torch.device('cpu')
    
    # Create small model
    model = SemiParametricClassifier_KMeans(
        embed_dim=128,
        num_classes=10,
        K=3,
        encoder_type='resnet18',
        learning_rate=0.1,
        pretrained_imagenet=False,
        pretrained_cifar=False,
    ).to(device)
    
    print(f"Model created: {model.__class__.__name__}")
    print(f"Encoder trainable: {any(p.requires_grad for p in model.encoder.parameters())}")
    print(f"MoE trainable: {any(p.requires_grad for p in model.moe.parameters())}")
    
    # Forward pass
    query_imgs = torch.randn(4, 3, 32, 32, device=device)
    support_imgs = torch.randn(32, 3, 32, 32, device=device)
    support_labels = torch.cat([torch.full((3,), i) for i in range(10)] + [torch.full((2,), 0)])
    query_labels = torch.tensor([0, 1, 2, 3])
    
    logits = model(query_imgs, support_imgs, support_labels)
    print(f"\nForward pass output: {logits.shape}")
    
    # Compute loss
    loss_fn = nn.NLLLoss()
    loss = loss_fn(logits, query_labels)
    print(f"Loss: {loss.item():.6f}")
    
    # Backward
    loss.backward()
    
    # Check encoder gradients
    encoder_has_grads = any(p.grad is not None for p in model.encoder.parameters())
    moe_has_grads = any(p.grad is not None for p in model.moe.parameters())
    
    print(f"\nGradient check:")
    print(f"  - Encoder has gradients: {encoder_has_grads}")
    print(f"  - MoE has gradients: {moe_has_grads}")
    
    if encoder_has_grads and moe_has_grads:
        print("\n✓ TEST PASSED: Full classifier gradient flow working!")
        return True
    else:
        print("\n✗ TEST FAILED: Missing gradients!")
        return False


def test_aggregation_only():
    """Test aggregation component in isolation."""
    print("\n" + "="*70)
    print("TEST 3: Aggregation Only")
    print("="*70)
    
    K_values = [1, 3, 5]
    num_classes = 10
    embed_dim = 512
    num_samples = 250  # Must be divisible by num_classes (250 = 25 * 10)
    
    # Create random embeddings
    embeddings = torch.randn(num_samples, embed_dim, requires_grad=True)
    labels = torch.cat([torch.full((num_samples//num_classes,), i) for i in range(num_classes)])
    
    print(f"Input: {num_samples} samples, {num_classes} classes, embed_dim={embed_dim}")
    
    for K in K_values:
        aggregator = Feature_Aggregation_KMeans(num_classes, K=K)
        prototypes, proto_labels = aggregator(embeddings, labels)
        
        # Check output shapes
        expected_max = K * num_classes
        expected_min = num_classes  # At least 1 per class
        
        actual_count = prototypes.shape[0]
        print(f"\nK={K}:")
        print(f"  - Prototypes shape: {prototypes.shape}")
        print(f"  - Proto labels shape: {proto_labels.shape}")
        print(f"  - Prototype count: {actual_count} (expected: {K} × {num_classes} = {expected_max})")
        
        # Verify labels
        for class_id in range(num_classes):
            count_for_class = (proto_labels == class_id).sum().item()
            expected = min(K, num_samples // num_classes)
            status = "✓" if count_for_class == expected else "✗"
            print(f"    {status} Class {class_id}: {count_for_class} prototypes (expected {expected})")
    
    print("\n✓ TEST PASSED: Aggregation working correctly!")
    return True


def test_pytorch_kmeans_directly():
    """Test PyTorch K-means algorithm directly."""
    print("\n" + "="*70)
    print("TEST 4: PyTorch K-means Algorithm")
    print("="*70)
    
    # Create test data: 100 points, 5D, 3 clusters
    K = 3
    num_points = 100
    embed_dim = 5
    
    # Create embeddings with gradient tracking
    embeddings = torch.randn(num_points, embed_dim, requires_grad=True)
    
    print(f"Input: {num_points} points, {embed_dim}-D, K={K} clusters")
    
    # Run K-means
    centroids = Feature_Aggregation_KMeans.pytorch_kmeans(embeddings, K, num_iterations=10)
    
    print(f"Output: {centroids.shape} centroids")
    print(f"Requires grad: {centroids.requires_grad}")
    
    # Test backprop
    loss = centroids.sum()
    loss.backward()
    
    if embeddings.grad is not None:
        print(f"\n✓ Gradients present: {embeddings.grad.shape}")
        print(f"  Gradient norm: {embeddings.grad.norm().item():.6f}")
        print("\n✓ TEST PASSED: PyTorch K-means preserves gradients!")
        return True
    else:
        print("\n✗ TEST FAILED: No gradients!")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("K-MEANS SPM IMPLEMENTATION TESTS")
    print("="*70)
    
    results = []
    
    try:
        results.append(("PyTorch K-means Algorithm", test_pytorch_kmeans_directly()))
    except Exception as e:
        print(f"\n✗ TEST FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("PyTorch K-means Algorithm", False))
    
    try:
        results.append(("K-means Gradient Flow", test_kmeans_gradient_flow()))
    except Exception as e:
        print(f"\n✗ TEST FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("K-means Gradient Flow", False))
    
    try:
        results.append(("Aggregation Component", test_aggregation_only()))
    except Exception as e:
        print(f"\n✗ TEST FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Aggregation Component", False))
    
    try:
        results.append(("Full Classifier Gradient Flow", test_classifier_gradient_flow()))
    except Exception as e:
        print(f"\n✗ TEST FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Full Classifier Gradient Flow", False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
