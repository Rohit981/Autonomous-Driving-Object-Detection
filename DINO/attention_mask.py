import torch
import torch.nn as nn

def build_attention_mask(
        num_dn_queries,
        device,
        num_queries
):
    total_queries = (
        num_dn_queries
        + num_queries
    )

    #False = attention allowed
    #True = attention blocked
    attn_mask = torch.zeros(
        total_queries,
        total_queries,
        dtype=torch.bool,
        device=device
    )

    #Block Matching queries from attending to DN queries

    attn_mask[
        num_dn_queries:,
        :num_dn_queries
    ]=True

    #Block DN queries from attending to matching queries
    attn_mask[
        :num_dn_queries,
        num_dn_queries:
    ]=True

    return attn_mask