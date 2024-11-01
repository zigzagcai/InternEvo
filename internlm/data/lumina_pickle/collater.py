import torch

def lumina_collate_fn(batch):
    """
    Collate function for packed input sequences.

    Args:
        batch (List[Dict]): List of dictionaries representing each sample in batch.
            Each dictionary contains "tokens", "labels", "type_ids", "cu_seqlens", and "indexes" keys.

    Returns:
        Tuple[Dict[str, torch.Tensor], torch.Tensor]: A tuple containing a dictionary of tensors with "input_ids",
            "cu_seqlens", "indexes", and "type_ids" keys, and the tensor of padded "labels".
    """
    # Initialize lists to store the data from each sample
    tokens, labels, type_ids, indexes = [], [], [], []
    cumulative_seqlens = [0]

    # Accumulate all samples into respective lists
    for sample in batch:
        tokens.extend([abs(w) for w in sample["tokens"]])
        labels.extend([w if w > 0 else -100 for w in sample["labels"]])
        type_ids.extend(sample["type_ids"])
        indexes.extend(sample["indexes"])
        cumulative_seqlens.append(cumulative_seqlens[-1] + sample["cu_seqlens"][-1])

    # Convert lists to tensors and unsqueeze to add batch dimension
    xs = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    ys = torch.tensor(labels, dtype=torch.long).unsqueeze(0)
    ts = torch.tensor(type_ids, dtype=torch.long).unsqueeze(0)
    indexes = torch.tensor(indexes, dtype=torch.long).unsqueeze(0)
    cu_seqlens = torch.tensor(cumulative_seqlens, dtype=torch.int).unsqueeze(0)

    return {"input_ids": xs, "cu_seqlens": cu_seqlens, "indexes": indexes, "type_ids": ts}, ys