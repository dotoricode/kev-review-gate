# SPDX-License-Identifier: Apache-2.0
import types
import pytest
from training_memory import admission_bytes, restore_optimizer, violation, GIB


def test_original_memory_time_and_swap_limits_are_not_relaxed(tmp_path):
    memory={'available':4*GIB,'swap_delta':512*1024**2}
    assert violation(memory,900) is None
    assert violation(dict(memory,available=4*GIB-1),900)=='available_memory_below_4_gib'
    assert violation(dict(memory,swap_delta=512*1024**2+1),900)=='swap_growth_above_512_mib'
    assert violation(memory,900.01)=='segment_above_900_seconds'
    (tmp_path/'base.safetensors').write_bytes(b'x'*16)
    (tmp_path/'resume.pt').write_bytes(b'x'*8)
    assert admission_bytes(tmp_path,tmp_path/'resume.pt')==16+4*GIB+8+512*1024**2


def test_mapped_resume_keeps_optimizer_and_rng_exact(tmp_path):
    # A tiny CPU optimizer comparison; this neither loads the Kev backbone nor allocates MPS tensors.
    import torch
    torch.manual_seed(722)
    model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    loss=model(torch.ones(1,2)).square().sum();loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    saved={'step':25,'protocol_sha256':'protocol','optimizer':optimizer.state_dict(),
           'cpu_rng':torch.get_rng_state(),'mps_rng':torch.tensor([1],dtype=torch.uint8)}
    path=tmp_path/'resume.pt';torch.save(saved,path)
    target=torch.nn.Linear(2,1);target.load_state_dict(model.state_dict())
    loaded=torch.optim.AdamW(target.parameters(),lr=2e-5,weight_decay=.01)
    seen=[];mapped=[]
    def load(*args,**kwargs):
        mapped.append(kwargs.get('mmap'));return torch.load(*args,**kwargs)
    api=types.SimpleNamespace(load=load,set_rng_state=torch.set_rng_state,mps=types.SimpleNamespace(set_rng_state=lambda x:seen.append(x)))
    restore_optimizer(api,loaded,path,25,'protocol')
    assert mapped==[True] and torch.equal(torch.get_rng_state(),saved['cpu_rng']) and torch.equal(seen[0],saved['mps_rng'])
    for a,b in zip(optimizer.state.values(),loaded.state.values()):
        assert all(torch.equal(a[k],b[k]) for k in a)
    for m,o in [(model,optimizer),(target,loaded)]:
        torch.set_rng_state(saved['cpu_rng']);o.zero_grad(set_to_none=True)
        m(torch.rand(1,2)).square().sum().backward();o.step()
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),target.parameters()))
    with pytest.raises(ValueError,match='resume_state_mismatch'):restore_optimizer(api,loaded,path,50,'protocol')
