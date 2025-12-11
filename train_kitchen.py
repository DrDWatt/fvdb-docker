#!/usr/bin/env python3
"""
Direct training script for kitchen scene - no containers needed
Run with: conda activate fvdb && python train_kitchen.py
"""

import sys
from pathlib import Path

# Data paths
dataset_path = Path("/home/dwatkins3/fvdb-docker/data/kitchen_scene")
output_path = Path("/home/dwatkins3/fvdb-docker/outputs")
output_path.mkdir(exist_ok=True, parents=True)

print("=" * 60)
print("🚀 GAUSSIAN SPLAT TRAINING - KITCHEN SCENE")
print("=" * 60)
print(f"\n📂 Dataset: {dataset_path}")
print(f"📂 Output: {output_path}")

# Check dataset exists
if not dataset_path.exists():
    print(f"\n❌ ERROR: Dataset not found at {dataset_path}")
    sys.exit(1)

# Check for COLMAP data
sparse_dir = dataset_path / "sparse" / "0"
if not sparse_dir.exists():
    print(f"\n❌ ERROR: COLMAP sparse reconstruction not found")
    print(f"   Expected: {sparse_dir}")
    sys.exit(1)

images_dir = dataset_path / "images"
num_images = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
print(f"📸 Images found: {num_images}")

if num_images == 0:
    print("\n❌ ERROR: No images found")
    sys.exit(1)

print("\n🔧 Loading fVDB Reality Capture...")
try:
    import fvdb
    import fvdb_reality_capture as frc
    print("✅ fVDB loaded successfully")
except ImportError as e:
    print(f"\n❌ ERROR: Failed to import fVDB: {e}")
    print("\nMake sure you're in the fvdb conda environment:")
    print("  conda activate fvdb")
    sys.exit(1)

print("\n📊 Loading COLMAP scene...")
try:
    scene = frc.sfm_scene.SfmScene.from_colmap(str(dataset_path))
    print(f"✅ Loaded scene: {len(scene.images)} images, {len(scene.points)} points")
except Exception as e:
    print(f"\n❌ ERROR: Failed to load COLMAP scene: {e}")
    sys.exit(1)

# Training configuration
num_steps = 30000
steps_per_epoch = len(scene.images)
total_epochs = int(num_steps / steps_per_epoch)
refine_until = int(total_epochs * 0.95)

print(f"\n🎯 Training Configuration:")
print(f"   • Steps: {num_steps}")
print(f"   • Epochs: {total_epochs}")
print(f"   • Refine until epoch: {refine_until}")
print(f"   • Images per epoch: {steps_per_epoch}")

config = frc.radiance_fields.GaussianSplatReconstructionConfig(
    max_steps=num_steps,
    refine_stop_epoch=refine_until,
    refine_every_epoch=0.5
)

print("\n🔨 Creating Gaussian Splat reconstruction...")
runner = frc.radiance_fields.GaussianSplatReconstruction.from_sfm_scene(
    scene,
    config=config
)

print("\n🏋️  Starting training...")
print(f"   This will take approximately 25-35 minutes")
print(f"   Progress will be shown below:\n")

try:
    runner.optimize()
    model = runner.model
    print("\n✅ Training complete!")
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Save model
output_file = output_path / "kitchen_gaussian_splat.ply"
print(f"\n💾 Saving model to: {output_file}")
try:
    model.save_ply(str(output_file), metadata=runner.reconstruction_metadata)
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✅ Model saved: {file_size_mb:.1f} MB")
except Exception as e:
    print(f"\n❌ Failed to save model: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("🎉 SUCCESS - TRAINING COMPLETE!")
print("=" * 60)
print(f"\n📂 Output: {output_file}")
print(f"📊 Model size: {file_size_mb:.1f} MB")
print(f"\n✨ Next steps:")
print(f"   • View in USD Pipeline: http://localhost:8002")
print(f"   • Stream via WebRTC: http://localhost:8080/test")
print(f"   • Or copy to models/: cp {output_file} /home/dwatkins3/fvdb-docker/data/models/")
print()
