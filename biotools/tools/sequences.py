import random
import torch

def sample_subsequences(sequence, n, min_length, max_length):
    sequence_length = len(sequence)

    if max_length is None:
        max_length = sequence_length

    if sequence_length <= min_length:
        return []

    sampled_subsequences = []

    for _ in range(n):
        start_index = torch.randint(0, sequence_length - min_length + 1, (1,)).item()
        max_length = min(sequence_length - start_index, max_length)
        substring_length = torch.randint(min_length, max_length + 1, (1,)).item()
        sampled_subsequences.append(
            sequence[start_index : start_index + substring_length]
        )

    return sampled_subsequences


def generate_protein_sequence(length):
    """
    Generates a random protein sequence of a given length.

    Args:
        length (int): Length of the protein sequence.

    Returns:
        str: Randomly generated protein sequence.
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # Standard 20 amino acid codes
    return "".join(random.choices(amino_acids, k=length))


def generate_multiple_protein_sequences(num_sequences, length_range):
    """
    Generates multiple random protein sequences of varying lengths.

    Args:
        num_sequences (int): Number of protein sequences to generate.
        length_range (tuple): A tuple specifying the min and max length of sequences (inclusive).

    Returns:
        list: A list of random protein sequences.
    """
    sequences = []
    for _ in range(num_sequences):
        length = random.randint(*length_range)
        sequences.append(generate_protein_sequence(length))
    return sequences