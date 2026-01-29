"""
DNA Tokenizer for genomic sequence modeling.

Single nucleotide character-level tokenizer for DNA sequences.
Vocabulary is aligned with Megatron-LM's DNANucleotideTokenizer.

Vocabulary (16 tokens):
- 0: [PAD] - Padding token
- 1: [UNK] - Unknown token
- 2: [CLS] - Classification token
- 3: [SEP] - Separator token  
- 4: [MASK] - Mask token for MLM
- 5: [BOS] - Beginning of sequence
- 6: [EOS] - End of sequence / End of document
- 7: A - Adenine
- 8: T - Thymine
- 9: C - Cytosine
- 10: G - Guanine
- 11-15: [RESERVED] - Reserved for future use
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import AddedToken, BatchEncoding


class DNATokenizer(PreTrainedTokenizer):
    """
    Single nucleotide tokenizer for DNA sequences.
    
    This tokenizer is designed for DNA sequences containing only A, C, T, G.
    It uses a vocabulary of 16 tokens (padded for GPU efficiency) that is
    compatible with Megatron-LM's DNANucleotideTokenizer.
    
    Vocabulary:
    - 0: [PAD] - Padding token
    - 1: [UNK] - Unknown token
    - 2: [CLS] - Classification token (start of sequence)
    - 3: [SEP] - Separator token
    - 4: [MASK] - Mask token for MLM
    - 5: [BOS] - Beginning of sequence
    - 6: [EOS] - End of sequence / End of document
    - 7: A - Adenine
    - 8: T - Thymine
    - 9: C - Cytosine
    - 10: G - Guanine
    - 11-15: [RESERVED] - Reserved for future use
    """
    
    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]
    
    # DNA nucleotides (only canonical bases, no N)
    DNA_CHARS = ["A", "T", "C", "G"]
    
    # Special tokens (matches Megatron DNANucleotideTokenizer)
    SPECIAL_TOKENS = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
        "[BOS]": 5,
        "[EOS]": 6,
    }
    
    # Nucleotide token IDs (matches Megatron DNANucleotideTokenizer)
    NUCLEOTIDE_IDS = {
        "A": 7,
        "T": 8,
        "C": 9,
        "G": 10,
    }
    
    def __init__(
        self,
        model_max_length: int = 512,
        padding_side: str = "right",
        add_special_tokens: bool = True,
        **kwargs
    ):
        """
        Initialize the DNA tokenizer.
        
        Args:
            model_max_length: Maximum sequence length (default: 512)
            padding_side: Side to pad ('left' or 'right')
            add_special_tokens: Whether to add CLS/SEP tokens
        """
        self.model_max_length = model_max_length
        self._add_special_tokens = add_special_tokens
        
        # Build vocabulary
        self._vocab_str_to_int = {}
        self._vocab_str_to_int.update(self.SPECIAL_TOKENS)
        self._vocab_str_to_int.update(self.NUCLEOTIDE_IDS)
        
        # Add reserved tokens to pad to 16
        for i in range(11, 16):
            self._vocab_str_to_int[f"[RESERVED_{i}]"] = i
        
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        
        # Define special tokens
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        mask_token = AddedToken("[MASK]", lstrip=False, rstrip=False)
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[EOS]", lstrip=False, rstrip=False)
        
        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            cls_token=cls_token,
            sep_token=sep_token,
            mask_token=mask_token,
            bos_token=bos_token,
            eos_token=eos_token,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )
    
    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size (padded to 16)."""
        return 16
    
    @property
    def eod_id(self) -> int:
        """End of document token ID (same as EOS)."""
        return self.SPECIAL_TOKENS["[EOS]"]
    
    def get_vocab(self) -> Dict[str, int]:
        """Return vocabulary as a dictionary."""
        return self._vocab_str_to_int.copy()
    
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize a DNA string into characters.
        
        Converts to uppercase. Unknown characters are mapped to [UNK].
        """
        text = text.upper()
        tokens = []
        for char in text:
            if char in self.DNA_CHARS:
                tokens.append(char)
            elif char in [" ", "\n", "\t"]:
                # Skip whitespace
                continue
            else:
                # Unknown character - skip or could use [UNK]
                # For DNA-only data, we skip unknown characters
                continue
        return tokens
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token to its ID."""
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert an ID to its token."""
        return self._vocab_int_to_str.get(index, "[UNK]")
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a sequence of tokens to a single DNA string."""
        # Filter out special tokens
        dna_tokens = [t for t in tokens if t in self.DNA_CHARS]
        return "".join(dna_tokens)
    
    def build_inputs_with_special_tokens(
        self, 
        token_ids_0: List[int], 
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs by adding special tokens.
        
        Format: [CLS] sequence [SEP]
        """
        if not self._add_special_tokens:
            return token_ids_0
        
        cls_id = [self._vocab_str_to_int["[CLS]"]]
        sep_id = [self._vocab_str_to_int["[SEP]"]]
        
        if token_ids_1 is None:
            return cls_id + token_ids_0 + sep_id
        return cls_id + token_ids_0 + sep_id + token_ids_1 + sep_id
    
    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """Get mask identifying special tokens."""
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )
        
        if not self._add_special_tokens:
            return [0] * len(token_ids_0)
        
        result = [1] + [0] * len(token_ids_0) + [1]
        if token_ids_1 is not None:
            result += [0] * len(token_ids_1) + [1]
        return result
    
    def create_token_type_ids_from_sequences(
        self, 
        token_ids_0: List[int], 
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Create token type IDs (all zeros for single sequence)."""
        if not self._add_special_tokens:
            return [0] * len(token_ids_0)
        
        result = [0] * (len(token_ids_0) + 2)  # CLS + sequence + SEP
        if token_ids_1 is not None:
            result += [1] * (len(token_ids_1) + 1)  # sequence + SEP
        return result
    
    def save_vocabulary(
        self, 
        save_directory: str, 
        filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        """Save the tokenizer vocabulary."""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._vocab_str_to_int, f, ensure_ascii=False, indent=2)
        
        return (vocab_file,)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load tokenizer from a directory."""
        vocab_file = os.path.join(pretrained_model_name_or_path, "vocab.json")
        if os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            # Extract model_max_length from kwargs or use default
            model_max_length = kwargs.pop("model_max_length", 512)
            return cls(model_max_length=model_max_length, **kwargs)
        return cls(**kwargs)
    
    def __call__(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_attention_mask: Optional[bool] = True,
        **kwargs
    ) -> BatchEncoding:
        """
        Main tokenization method.
        
        Args:
            text: DNA sequence(s) to tokenize
            add_special_tokens: Whether to add CLS/SEP tokens
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum length (default: model_max_length)
            return_tensors: Return format ("pt" for PyTorch, "np" for NumPy)
            return_attention_mask: Whether to return attention mask
        
        Returns:
            BatchEncoding with input_ids and attention_mask
        """
        # Handle single string vs list
        if isinstance(text, str):
            text = [text]
        
        max_length = max_length or self.model_max_length
        
        # Store original setting and override if specified
        orig_add_special = self._add_special_tokens
        self._add_special_tokens = add_special_tokens
        
        # Tokenize all sequences
        input_ids_list = []
        for seq in text:
            tokens = self._tokenize(seq)
            token_ids = [self._convert_token_to_id(t) for t in tokens]
            
            # Add special tokens if requested
            if add_special_tokens:
                token_ids = self.build_inputs_with_special_tokens(token_ids)
            
            # Truncate if needed
            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            
            input_ids_list.append(token_ids)
        
        # Restore original setting
        self._add_special_tokens = orig_add_special
        
        # Apply padding
        if padding:
            max_len = max(len(ids) for ids in input_ids_list)
            if padding == "max_length" and max_length:
                max_len = max_length
            
            padded_input_ids = []
            attention_mask = []
            
            for ids in input_ids_list:
                padding_length = max_len - len(ids)
                
                if self.padding_side == "right":
                    padded_ids = ids + [self.pad_token_id] * padding_length
                    mask = [1] * len(ids) + [0] * padding_length
                else:
                    padded_ids = [self.pad_token_id] * padding_length + ids
                    mask = [0] * padding_length + [1] * len(ids)
                
                padded_input_ids.append(padded_ids)
                attention_mask.append(mask)
            
            input_ids_list = padded_input_ids
        else:
            attention_mask = [[1] * len(ids) for ids in input_ids_list]
        
        # Build result
        result = {"input_ids": input_ids_list}
        if return_attention_mask:
            result["attention_mask"] = attention_mask
        
        # Convert to tensors
        if return_tensors == "pt":
            result = {k: torch.tensor(v) for k, v in result.items()}
        elif return_tensors == "np":
            result = {k: np.array(v) for k, v in result.items()}
        
        return BatchEncoding(result, tensor_type=return_tensors)
    
    def decode(
        self,
        token_ids: Union[int, List[int], torch.Tensor, np.ndarray],
        skip_special_tokens: bool = False,
        **kwargs
    ) -> str:
        """Decode token IDs to a DNA string."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        
        tokens = [self._convert_id_to_token(tid) for tid in token_ids]
        
        if skip_special_tokens:
            special_tokens = set(self.all_special_tokens)
            tokens = [t for t in tokens if t not in special_tokens]
        
        return self.convert_tokens_to_string(tokens)
    
    def batch_decode(
        self,
        sequences: Union[List[List[int]], torch.Tensor, np.ndarray],
        skip_special_tokens: bool = False,
        **kwargs
    ) -> List[str]:
        """Decode a batch of token IDs."""
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        elif isinstance(sequences, np.ndarray):
            sequences = sequences.tolist()
        
        return [
            self.decode(seq, skip_special_tokens=skip_special_tokens, **kwargs)
            for seq in sequences
        ]
    
    def encode_dna(self, sequence: str, add_eos: bool = False) -> List[int]:
        """
        Simple DNA sequence encoding (for Megatron compatibility).
        
        Args:
            sequence: DNA sequence string
            add_eos: Whether to append EOS token
        
        Returns:
            List of token IDs
        """
        ids = []
        for char in sequence.upper():
            if char in self.NUCLEOTIDE_IDS:
                ids.append(self.NUCLEOTIDE_IDS[char])
        
        if add_eos:
            ids.append(self.eod_id)
        
        return ids


def create_megatron_tokenizer(model_max_length: int = 512) -> DNATokenizer:
    """
    Create a DNA tokenizer configured for Megatron-LM.
    
    Args:
        model_max_length: Maximum sequence length
    
    Returns:
        Configured DNATokenizer instance
    """
    return DNATokenizer(
        model_max_length=model_max_length,
        padding_side="right",
        add_special_tokens=False,  # Megatron handles this
    )


if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = DNATokenizer(model_max_length=512)
    
    # Test basic tokenization
    sequence = "ATCGATCGATCG"
    encoded = tokenizer(
        sequence,
        add_special_tokens=True,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_tensors="pt"
    )
    
    print("=" * 60)
    print("DNA Tokenizer Test")
    print("=" * 60)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Vocabulary: {tokenizer.get_vocab()}")
    print(f"\nInput sequence: {sequence}")
    print(f"Token IDs: {encoded['input_ids'].tolist()}")
    print(f"Attention mask: {encoded['attention_mask'].tolist()}")
    print(f"Decoded: {tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)}")
    
    # Test simple encoding (Megatron style)
    print(f"\nSimple encode (no special tokens):")
    ids = tokenizer.encode_dna(sequence, add_eos=True)
    print(f"  Token IDs: {ids}")
    print(f"  Expected: A=7, T=8, C=9, G=10, EOS=6")
