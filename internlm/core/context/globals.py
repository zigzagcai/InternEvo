import torch


class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance


class ProcessGroupSingleton(Singleton):
    def __init__(self):
        self.ULYSSES_PG = None
        self.RING_PG = None
        self.ALLGATHER_PG = None
        self.P2P_PG = None

        self.ULYSSES_TUTEL_INTRA_PG = None
        self.ULYSSES_TUTEL_INTER_PG = None


PROCESS_GROUP = ProcessGroupSingleton()


def get_sliding_window_pg(rank, window_num, window_size, ring_ranks, interleaved):
    
    all_gather_pg, p2p_pg = None, None
    
    if interleaved is False:
                    
        # ring_ranks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        # window_size = 4
        # window_num = 4
        # all_gather = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        # p2p = [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]
    
        # create the all-gather process group when using sliding window
        for j in range(window_num):
            window_ranks = ring_ranks[j * window_size : (j + 1) * window_size]
            # print(f'ring_ranks:{ring_ranks}')
            group = torch.distributed.new_group(window_ranks)
            if rank in window_ranks:
                all_gather_pg = group
                    
        # create the p2p process group when using sliding window
        for j in range(window_size):
            p2p_ranks = []
            for t in range(window_num):
                p2p_ranks.append(ring_ranks[t * window_size + j])
            group = torch.distributed.new_group(p2p_ranks)
            if rank in p2p_ranks:
                p2p_pg = group
        
    else:
        # ring_ranks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        # window_size = 4
        # window_num = 4
        # all_gather = [[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15]]
        # p2p = [[0, 1, 8, 9], [2, 3, 12, 13], [4, 5, 12, 13], [6, 7, 14, 15]]                   
        
        # create the all-gather process group when using sliding window
        even_ranks = []
        odd_ranks = []
        
        for r in ring_ranks:
            if r % 2 == 0:
                even_ranks.append(r)
            else:
                odd_ranks.append(r)
        assert len(even_ranks) == len(odd_ranks)
        start_ranks = []
        for i in range(len(even_ranks)):
            if i % window_size == 0:
                start_ranks.append(even_ranks[i])
                start_ranks.append(odd_ranks[i])
        
        for start_rank in start_ranks:
            window_ranks = []
            offset = start_rank
            for i in range(window_size):
                window_ranks.append(offset)
                offset += 2
            group = torch.distributed.new_group(window_ranks)
            if rank in window_ranks:
                all_gather_pg = group
        
        # create the p2p process group when using sliding window
        for i in range(window_size):
            p2p_ranks = []
            for j in range(len(even_ranks)):
                if j % window_size == i:
                    p2p_ranks.append(even_ranks[j])
                    p2p_ranks.append(odd_ranks[j])
            
            group = torch.distributed.new_group(p2p_ranks)
            if rank in p2p_ranks:
                p2p_pg = group
    
    return all_gather_pg, p2p_pg


def get_tutel_gp(rank, sp_ulysses_degree, sp_ring_degree, ulysses_ranks_all, my_ulysses_idx, allow_pure_intra_tutel, allow_pure_inter_tutel, use_ulysses_low):
    if sp_ulysses_degree <=8 and use_ulysses_low:
        if allow_pure_intra_tutel:
            intra_size = sp_ulysses_degree // 2
        else:
            return None, None
    
    if not use_ulysses_low and sp_ring_degree >=8:
        if allow_pure_inter_tutel:
            intra_size = sp_ulysses_degree // 2
        else:
            return None, None

    inter_size = sp_ulysses_degree // intra_size
    # local_ulysses_rank = rank % sp_ulysses_degree
    # my_ulysses_rank_idx = rank // sp_ulysses_degree 
    
    for j in range(len(ulysses_ranks_all)):
        ulysses_ranks = ulysses_ranks_all[j]
        ulysses_ranks = torch.tensor(ulysses_ranks, dtype=torch.int64)
        my_ulysses_local_rank = rank % sp_ulysses_degree
        
        inter_all_ranks = ulysses_ranks.reshape(inter_size, intra_size)
        for i in range(inter_size):
            intra_pg = inter_all_ranks[i].tolist()
            intra_pg = torch.distributed.new_group(intra_pg)
            if j == my_ulysses_idx and i == my_ulysses_local_rank // intra_size:
                tutel_intra_group = intra_pg
                print(f"Rank: {rank} intra pg: {inter_all_ranks[i].tolist()}", flush=True)

        intra_all_ranks = ulysses_ranks.reshape(inter_size, intra_size).T
        for i in range(intra_size):
            inter_pg = intra_all_ranks[i].tolist()
            inter_pg = torch.distributed.new_group(inter_pg)
            if j == my_ulysses_idx and i == my_ulysses_local_rank % intra_size:
                tutel_inter_group = inter_pg
                print(f"Rank: {rank} inter pg: {intra_all_ranks[i].tolist()}", flush=True)

    return tutel_intra_group, tutel_inter_group


def set_seq_parallel_pg(
    sp_ulysses_degree,
    sp_ring_degree,
    rank,
    world_size,
    use_ulysses_low=True,
    window_size=1,
    interleaved=False,
    use_single_all2all=True,
    use_tutel_all2all=False,
    allow_pure_intra_tutel=False,
    allow_pure_inter_tutel=False,
):
    """
    sp_ulysses_degree x sp_ring_degree = seq_parallel_degree
    (ulysses_degree, dp_degree)
    """
    sp_degree = sp_ring_degree * sp_ulysses_degree
    dp_degree = world_size // sp_degree

    assert world_size % sp_degree == 0, f"world_size {world_size} % sp_degree {sp_ulysses_degree} == 0"

    num_ulysses_pgs = sp_ring_degree  # world_size // sp_ulysses_degree
    num_ring_pgs = sp_ulysses_degree  #
    
    if window_size == sp_ring_degree or window_size >= 8:
        interleaved = False

    tutel_intra_group, tutel_inter_group = None, None
    ulysses_ranks_all = []
    if use_ulysses_low:
        for dp_rank in range(dp_degree):
            offset_dp = dp_rank * sp_degree
            for i in range(num_ulysses_pgs):
                ulysses_ranks = list(
                    range(
                        i * sp_ulysses_degree + offset_dp,
                        (i + 1) * sp_ulysses_degree + offset_dp,
                    )
                )
                # print(f'ulysses_ranks:{ulysses_ranks}')
                group = torch.distributed.new_group(ulysses_ranks)
                ulysses_ranks_all.append(ulysses_ranks)
                if rank in ulysses_ranks:
                    ulyssess_pg = group
                    my_ulysses_idx = i

            for i in range(num_ring_pgs):
                
                assert sp_ring_degree % window_size == 0, "the window size should be divided by the sp_ring_degree."
                window_num = sp_ring_degree // window_size
                
                ring_ranks = list(range(i + offset_dp, sp_degree + offset_dp, num_ring_pgs))
                group = torch.distributed.new_group(ring_ranks)
                if rank in ring_ranks:
                    ring_pg = group
                
                all_gather, p2p = get_sliding_window_pg(rank, window_num, window_size, ring_ranks, interleaved)
                if all_gather is not None:
                    all_gather_pg = all_gather
                if p2p is not None:
                    p2p_pg = p2p
    else:
        for dp_rank in range(dp_degree):
            offset_dp = dp_rank * sp_degree
            for i in range(num_ring_pgs):
                ring_ranks = list(range(i * sp_ring_degree + offset_dp, (i + 1) * sp_ring_degree + offset_dp))
                group = torch.distributed.new_group(ring_ranks)
                if rank in ring_ranks:
                    ring_pg = group
         
                assert sp_ring_degree % window_size == 0, "the window size should be divided by the sp_ring_degree."
                window_num = sp_ring_degree // window_size
                
                all_gather, p2p = get_sliding_window_pg(rank, window_num, window_size, ring_ranks, interleaved)
                if all_gather is not None:
                    all_gather_pg = all_gather
                if p2p is not None:
                    p2p_pg = p2p
                
            for i in range(num_ulysses_pgs):
                ulysses_ranks = list(range(i + offset_dp, sp_degree + offset_dp, num_ulysses_pgs))
                group = torch.distributed.new_group(ulysses_ranks)
                ulysses_ranks_all.append(ulysses_ranks)
                if rank in ulysses_ranks:
                    ulyssess_pg = group
                    my_ulysses_idx = i

    if use_tutel_all2all:
        assert use_single_all2all is False
        tutel_intra_group, tutel_inter_group = get_tutel_gp(
            rank,
            sp_ulysses_degree,
            sp_ring_degree,
            ulysses_ranks_all,
            my_ulysses_idx,
            allow_pure_intra_tutel,
            allow_pure_inter_tutel,
            use_ulysses_low,
        )

    PROCESS_GROUP.ULYSSES_PG = ulyssess_pg
    PROCESS_GROUP.ULYSSES_TUTEL_INTRA_PG = tutel_intra_group
    PROCESS_GROUP.ULYSSES_TUTEL_INTER_PG = tutel_inter_group
    PROCESS_GROUP.RING_PG = ring_pg
    PROCESS_GROUP.ALLGATHER_PG = all_gather_pg
    PROCESS_GROUP.P2P_PG = p2p_pg
