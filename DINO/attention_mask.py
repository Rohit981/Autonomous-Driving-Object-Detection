import torch
import torch.nn as nn

def build_attention_mask(
        num_dn_queries,
        device,
        num_queries,
        num_dn_groups
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

    #DN <-> Matching isolation

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

    #DN group isolation
    if num_dn_queries > 0:
        
        assert (
            num_dn_queries
            % num_dn_groups
            ==0
        ), (
            "num_dn_queries must be divisible"
            "by num_dn_groups"
        )

        #Queries per DN group
        queries_per_group = (
            num_dn_queries
            //num_dn_groups
        )

        for group_id in range(
            num_dn_groups
        ):
            group_start = (
                group_id
                * queries_per_group
            )

            group_end = (
                (group_id + 1)
                * queries_per_group
            )

            #Block this group from all OTHER DN queries
            attn_mask[
                group_start:group_end,
                :group_start
            ] = True

            attn_mask[
                group_start:group_end,
                group_end:num_dn_queries
            ] = True

    return attn_mask