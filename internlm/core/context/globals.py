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



def set_seq_parallel_pg(sp_ulysses_degree, sp_ring_degree, rank, world_size, use_ulysses_low=True, window_size=1, interleaved=False):
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
                if rank in ulysses_ranks:
                    ulyssess_pg = group

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
                if rank in ulysses_ranks:
                    ulyssess_pg = group

    PROCESS_GROUP.ULYSSES_PG = ulyssess_pg
    PROCESS_GROUP.RING_PG = ring_pg
    PROCESS_GROUP.ALLGATHER_PG = all_gather_pg
    PROCESS_GROUP.P2P_PG = p2p_pg
