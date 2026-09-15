import flashinfer
import pytest
import torch


def _reference_append(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    append_indptr: torch.Tensor,
    paged_kv_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    out = paged_kv_cache.clone()
    seq_lens = flashinfer.get_seq_lens(kv_indptr, kv_last_page_len, page_size)
    for seq_id in range(kv_last_page_len.numel()):
        start = int(append_indptr[seq_id].item())
        end = int(append_indptr[seq_id + 1].item())
        append_len = end - start
        seq_len = int(seq_lens[seq_id].item())
        base_pos = seq_len - append_len
        for token_i in range(append_len):
            pos = base_pos + token_i
            page_in_seq = pos // page_size
            offset_in_page = pos % page_size
            page_id = int(kv_indices[int(kv_indptr[seq_id].item()) + page_in_seq].item())
            token_id = start + token_i
            out[page_id, 0, offset_in_page] = append_key[token_id]
            out[page_id, 1, offset_in_page] = append_value[token_id]
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flashinfer_append_signature_shim_equivalent_kv_cache():
    torch.manual_seed(0)
    device = "cuda"
    page_size, num_kv_heads, head_dim = 16, 4, 128
    append_indptr = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
    kv_indices = torch.tensor([3, 7, 11, 13, 17], dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor([4, 9], dtype=torch.int32, device=device)
    nnz = int(append_indptr[-1].item())
    append_key = torch.randn(nnz, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    append_value = torch.randn(nnz, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    cache_a = torch.zeros(24, 2, page_size, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    cache_b = cache_a.clone()

    batch_indices, positions = flashinfer.get_batch_indices_positions(
        append_indptr, flashinfer.get_seq_lens(kv_indptr, kv_last_page_len, page_size), nnz
    )
    flashinfer.append_paged_kv_cache(
        append_key,
        append_value,
        batch_indices,
        positions,
        cache_a,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        kv_layout="NHD",
    )
    cache_b = _reference_append(
        append_key,
        append_value,
        append_indptr,
        cache_b,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        page_size,
    )
    torch.testing.assert_close(cache_a, cache_b)
