import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recogdrive"))

from navsim.agents.recogdrive.drivemem import (  # noqa: E402
    DriveMemModule,
    InvariantFeatureAdapter,
    MemoryFusionGate,
    PrototypeMemoryBank,
    TokenDSU,
)


def test_token_dsu_shape_and_eval_identity():
    x = torch.randn(4, 16, 32)
    dsu = TokenDSU(p=1.0, factor=0.5)
    dsu.train()
    y = dsu(x)
    assert y.shape == x.shape
    dsu.eval()
    y_eval = dsu(x)
    assert torch.equal(y_eval, x)


def test_ifa_shapes():
    x = torch.randn(4, 16, 32)
    ifa = InvariantFeatureAdapter(input_dim=32, ifa_dim=12, hidden_dim=64)
    z_inv, z_spu = ifa(x)
    assert z_inv.shape == (4, 12)
    assert z_spu.shape == (4, 12)


def test_pmb_retrieval_and_loss():
    q = torch.randn(4, 12)
    pmb = PrototypeMemoryBank(num_prototypes=8, memory_dim=12, top_k=3)
    z_mem, a, logs = pmb(q)
    assert z_mem.shape == q.shape
    assert a.shape == (4, 8)
    assert logs["topk_indices"].shape == (4, 3)
    loss, loss_logs = pmb.loss(q, z_mem, a)
    assert loss.ndim == 0
    assert "memory_alpha" in loss_logs


def test_memory_fusion_gate_shape():
    z_plan = torch.randn(4, 10, 32)
    z_mem = torch.randn(4, 12)
    fusion = MemoryFusionGate(planner_dim=32, memory_dim=12)
    z_fused, logs = fusion(z_plan, z_mem)
    assert z_fused.shape == z_plan.shape
    assert "memory_gate_mean" in logs


def test_drivemem_module_full_path():
    z_plan = torch.randn(4, 10, 32)
    module = DriveMemModule(
        input_dim=32,
        planner_dim=32,
        cfg={
            "enabled": True,
            "use_dsu_side_branch": True,
            "dsu_p": 1.0,
            "ifa_dim": 12,
            "ifa_hidden_dim": 64,
            "num_prototypes": 8,
            "top_k": 3,
        },
    )
    module.train()
    z_fused, losses, logs = module(z_plan)
    assert z_fused.shape == z_plan.shape
    assert "loss_memory_recon" in losses
    assert "assignment_entropy" in logs
