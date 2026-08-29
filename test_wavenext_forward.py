import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

from src.models.wavenext.gen import WaveNeXtGenerator
from src.data.qxs_saropt.datamodule import QXSSAROPTDataModule

CKPT = r"C:\Users\Sakshi Tiwari\Desktop\SAR\ANIX-SAR\WaveNeXt\generator.safetensors"
DATA = r"C:\Users\Sakshi Tiwari\Desktop\SAR\QXSLAB_SAROPT"
CONFIG = r".\src\models\wavenext\config.yaml"

print("=" * 60)
print("ANIX-SAR — WaveNeXt Forward Pass Test")
print("=" * 60)

# ------------------------------------------------------------
# 1. Load repository configuration
# ------------------------------------------------------------
print("\n[1] Loading WaveNeXt configuration...")
cfg = OmegaConf.load(CONFIG)

print("Config loaded:", CONFIG)

# ------------------------------------------------------------
# 2. Build generator using the REAL repository API
# ------------------------------------------------------------
print("\n[2] Building WaveNeXt generator...")
G = WaveNeXtGenerator(cfg)

print("Generator built successfully.")

# ------------------------------------------------------------
# 3. Load original pretrained checkpoint
# ------------------------------------------------------------
print("\n[3] Loading pretrained generator checkpoint...")

state = load_file(CKPT, device="cpu")

missing, unexpected = G.load_state_dict(state, strict=False)

print("Checkpoint tensors:", len(state))
print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

if missing:
    print("Missing:", missing)

if unexpected:
    print("Unexpected:", unexpected)

assert len(missing) == 0, "Checkpoint has missing keys."
assert len(unexpected) == 0, "Checkpoint has unexpected keys."

print("Checkpoint MATCH: OK")

# ------------------------------------------------------------
# 4. Load one real QXS-SAROPT batch
# ------------------------------------------------------------
print("\n[4] Loading one QXS-SAROPT batch...")

dm = QXSSAROPTDataModule(
    data_dir=DATA,
    batch_size=1,
    val_batch_size=1,
    image_size=256,
    num_workers=0,
    train_val_split_ratio=0.8,
    seed=42,
    sar_channels=1,
    use_augmentation=False,
    sar_subdir="sar_256_oc_0.2",
    opt_subdir="opt_256_oc_0.2",
)

dm.setup()

sar, opt = next(iter(dm.train_dataloader()))

print("SAR shape:", sar.shape)
print("OPT shape:", opt.shape)

# ------------------------------------------------------------
# 5. Move model to GPU
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n[5] Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM GB:",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2
        )
    )

G = G.to(device)
G.eval()

sar = sar.to(device)

# ------------------------------------------------------------
# 6. Run forward pass
# ------------------------------------------------------------
print("\n[6] Running forward pass...")

with torch.no_grad():
    fake = G(sar)

print("Output shape:", fake.shape)
print("Output min:", float(fake.min()))
print("Output max:", float(fake.max()))
print("Output mean:", float(fake.mean()))

# ------------------------------------------------------------
# 7. Validate output
# ------------------------------------------------------------
assert fake.ndim == 4
assert fake.shape == (1, 3, 256, 256)
assert torch.isfinite(fake).all()

print("\n" + "=" * 60)
print("FORWARD PASS OK")
print("=" * 60)
print("Input : [1, 1, 256, 256]")
print("Output: [1, 3, 256, 256]")
print("Pretrained WaveNeXt successfully processed real SAR data.")
print("=" * 60)
