import os
import urllib.request

import tabix
import pyBigWig
import pyfaidx
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ._base import register_dataset


# `MemmapGenome` and `GenomicSignalFeatures` are modified from `selene`.
# https://github.com/FunctionLab/selene

class MemmapGenome:
    """
    Memmapped version of selene.sequence.Genome. Faster for sequence
    retrieval by storing all precomputed one-hot encodings in a memmapped
    file (~40G for human genome).

    The memmapfile can be an exisiting memmapped file or a path where you
    want to create the memmapfile. If the specified memmapfile does not
    exist, it will be created the first time you call any method of
    MemmapGenome or if MemmapGenome is initialized with `init_unpickable=True`.
    Therefore the first call will take some time for the
    creation of memmapfile if it does not exist. Also,  if
    memmapfile has not been created, be careful not to run multiple
    instances of MemmapGenome in parallel (such as with Dataloader),
    because as each process will try to create the file.

    Parameters
    ----------
    input_path : str
        Path to an indexed FASTA file, that is, a `*.fasta` file with
        a corresponding `*.fai` file in the same directory. This file
        should contain the target organism's genome sequence.
    init_unpickleable : bool, optional
        Default is False. If False, delay part of initialization code
        to executed only when a relevant method is called. This enables
        the object to be pickled after instantiation. `init_unpickleable` should
        be `False` when used when multi-processing is needed e.g. DataLoader.
    memmapfile : str or None, optional
        Specify the numpy.memmap file for storing the encoding
        of the genome. If memmapfile does not exist, it will be
        created when the encoding is requested for the first time.

    Attributes
    ----------
    genome : pyfaidx.Fasta
        The FASTA file containing the genome sequence.
    chrs : list(str)
        The list of chromosome names.
    len_chrs : dict
        A dictionary mapping the names of each chromosome in the file to
        the length of said chromosome.
    """

    BASES_ARR = ['A', 'C', 'G', 'T']
    """
    This is an array with the alphabet (i.e. all possible symbols
    that may occur in a sequence). We expect that
    `INDEX_TO_BASE[i]==BASES_ARR[i]` is `True` for all valid `i`.

    """

    BASE_TO_INDEX = {
        'A': 0, 'C': 1, 'G': 2, 'T': 3,
        'a': 0, 'c': 1, 'g': 2, 't': 3,
    }
    """
    A dictionary mapping members of the alphabet (i.e. all
    possible symbols that can occur in a sequence) to integers.
    """

    INDEX_TO_BASE = {
        0: 'A', 1: 'C', 2: 'G', 3: 'T'
    }
    """
    A dictionary mapping integers to members of the alphabet (i.e.
    all possible symbols that can occur in a sequence). We expect
    that `INDEX_TO_BASE[i]==BASES_ARR[i]` is `True` for all
    valid `i`.
    """

    COMPLEMENTARY_BASE_DICT = {
        'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N',
        'a': 'T', 'c': 'G', 'g': 'C', 't': 'A', 'n': 'N'
    }
    """
    A dictionary mapping each base to its complementary base.
    """

    UNK_BASE = "N"
    """
    This is a base used to represent unknown positions. This is not
    the same as a character from outside the sequence's alphabet. A
    character from outside the alphabet is an error. A position with
    an unknown base signifies that the position is one of the bases
    from the alphabet, but we are uncertain which.
    """

    DOWNLOAD_URL = 'https://github.com/FunctionLab/selene/blob/master/selene_sdk/sequences/data/{}?raw=true'

    def __init__(
            self,
            root,
            input_path,
            memmapfile,
            blacklist_regions=None,
            bases_order=None,
            init_unpicklable=False
    ):
        self.root = root
        self.memmapfile = memmapfile
        self.input_path = input_path
        self.blacklist_regions = blacklist_regions
        self._initialized = False

        if bases_order is not None:
            bases = [str.upper(b) for b in bases_order]
            self.BASES_ARR = bases
            lc_bases = [str.lower(b) for b in bases]
            self.BASE_TO_INDEX = {
                **{b: ix for (ix, b) in enumerate(bases)},
                **{b: ix for (ix, b) in enumerate(lc_bases)}}
            self.INDEX_TO_BASE = {ix: b for (ix, b) in enumerate(bases)}
            self.update_bases_order(bases)

        if init_unpicklable:
            self._unpicklable_init()

    @classmethod
    def update_bases_order(cls, bases):
        cls.BASES_ARR = bases
        lc_bases = [str.lower(b) for b in bases]
        cls.BASE_TO_INDEX = {
            **{b: ix for (ix, b) in enumerate(bases)},
            **{b: ix for (ix, b) in enumerate(lc_bases)}}
        cls.INDEX_TO_BASE = {ix: b for (ix, b) in enumerate(bases)}

    def _unpicklable_init(self):
        if not self._initialized:
            self.genome = pyfaidx.Fasta(os.path.join(self.root, self.input_path))
            self.chrs = sorted(self.genome.keys())
            self.len_chrs = self._get_len_chrs()
            self._blacklist_tabix = None

            if self.blacklist_regions == "hg19":
                fn = os.path.join(self.root, "hg19_blacklist_ENCFF001TDO.bed.gz")
                if not os.path.isfile(fn):
                    urllib.request.urlretrieve(self.DOWNLOAD_URL.format("hg19_blacklist_ENCFF001TDO.bed.gz"), fn)
                self._blacklist_tabix = tabix.open(fn)
            elif self.blacklist_regions == "hg38":
                fn = os.path.join(self.root, "hg38.blacklist.bed.gz")
                if not os.path.isfile(fn):
                    urllib.request.urlretrieve(self.DOWNLOAD_URL.format("hg38.blacklist.bed.gz"), fn)
                self._blacklist_tabix = tabix.open(fn)
            elif self.blacklist_regions is not None:  # user-specified file
                self._blacklist_tabix = tabix.open(os.path.join(self.root, self.blacklist_regions))

            self.lens = np.array([self.len_chrs[c] for c in self.chrs])
            self.inds = {
                c: ind for c, ind in zip(self.chrs, np.concatenate([[0], np.cumsum(self.lens)]))
            }
            memmapfile = os.path.join(self.root, self.memmapfile)
            if os.path.isfile(memmapfile):
                # load memmap file
                self.sequence_data = np.memmap(memmapfile, dtype="float32", mode="r")
                self.sequence_data = np.reshape(
                    self.sequence_data, (4, int(self.sequence_data.shape[0] / 4))
                )
            else:
                # convert all sequences into encoding
                self.sequence_data = np.zeros((4, self.lens.sum()), dtype=np.float32)
                for c in self.chrs:
                    sequence = self.genome[c][:].seq
                    encoding = self.sequence_to_encoding(sequence)
                    self.sequence_data[:, self.inds[c]: self.inds[c] + self.len_chrs[c]] = encoding.T
                # create memmap file
                mmap = np.memmap(memmapfile, dtype="float32", mode="w+", shape=self.sequence_data.shape)
                mmap[:] = self.sequence_data
                self.sequence_data = np.memmap(memmapfile, dtype="float32", mode="r",
                                               shape=self.sequence_data.shape)

            self._initialized = True

    def get_encoding_from_coords(self, chrom, start, end, strand="+", pad=False):
        """
        Gets the one-hot encoding of the genomic sequence at the
        queried coordinates.

        Parameters
        ----------
        chrom : str
            The name of the chromosome or region, e.g. "chr1".
        start : int
            The 0-based start coordinate of the first position in the
            sequence.
        end : int
            One past the 0-based last position in the sequence.
        strand : {'+', '-', '.'}, optional
            Default is '+'. The strand the sequence is located on. '.' is
            treated as '+'.
        pad : bool, optional
            Default is `False`. Pad the output sequence with 'N' if `start`
            and/or `end` are out of bounds to return a sequence of length
            `end - start`.


        Returns
        -------
        numpy.ndarray, dtype=numpy.float32
            The :math:`L \\times 4` encoding of the sequence, where
            :math:`L = end - start`.

        Raises
        ------
        AssertionError
            If it cannot retrieve encoding that matches the length `L = end - start`
            such as when end > chromosome length and pad=False
        """
        self._unpicklable_init()
        if pad:
            # padding with 0.25 if coordinates extend beyond chr boundary
            if end > self.len_chrs[chrom]:
                pad_right = end - self.len_chrs[chrom]
                qend = self.len_chrs[chrom]
            else:
                qend = end
                pad_right = 0

            if start < 0:
                pad_left = 0 - start
                qstart = 0
            else:
                pad_left = 0
                qstart = start

            encoding = np.hstack(
                [
                    np.ones((4, pad_left)) * 0.25,
                    self.sequence_data[:, self.inds[chrom] + qstart: self.inds[chrom] + qend],
                    np.ones((4, pad_right)) * 0.25,
                ]
            )
        else:
            assert end <= self.len_chrs[chrom] and start >= 0
            encoding = self.sequence_data[:, self.inds[chrom] + start: self.inds[chrom] + end]

        if strand == "-":
            encoding = encoding[::-1, ::-1]
        assert encoding.shape[1] == end - start
        return encoding.T

    def get_chrs(self):
        """Gets the list of chromosome names.

        Returns
        -------
        list(str)
            A list of the chromosome names.

        """
        self._unpicklable_init()
        return self.chrs

    def get_chr_lens(self):
        """Gets the name and length of each chromosome sequence in the file.

        Returns
        -------
        list(tuple(str, int))
            A list of tuples of the chromosome names and lengths.

        """
        self._unpicklable_init()
        return [(k, self.len_chrs[k]) for k in self.get_chrs()]

    def _get_len_chrs(self):
        len_chrs = {}
        for chrom in self.chrs:
            len_chrs[chrom] = len(self.genome[chrom])
        return len_chrs

    def _genome_sequence(self, chrom, start, end, strand='+'):
        if strand == '+' or strand == '.':
            return self.genome[chrom][start:end].seq
        else:
            return self.genome[chrom][start:end].reverse.complement.seq

    @classmethod
    def sequence_to_encoding(cls, sequence):
        """Converts an input sequence to its one-hot encoding.

        Parameters
        ----------
        sequence : str
            A nucleotide sequence of length :math:`L`

        Returns
        -------
        numpy.ndarray, dtype=numpy.float32
            The :math:`L \\times 4` one-hot encoding of the sequence.

        """
        encoding = np.zeros((len(sequence), len(cls.BASES_ARR)), dtype=np.float32)
        for i, base in enumerate(sequence):
            if base in cls.BASE_TO_INDEX:
                encoding[i, cls.BASE_TO_INDEX[base]] = 1
            else:
                encoding[i, :] = 1 / len(cls.BASES_ARR)
        return encoding


class GenomicSignalFeatures:
    """
    #Accept a list of cooler files as input.
    """

    def __init__(self, input_paths, features, shape, blacklists=None, blacklists_indices=None,
                 replacement_indices=None, replacement_scaling_factors=None):
        """
        Constructs a new `GenomicFeatures` object.
        """
        self.input_paths = input_paths
        self.initialized = False
        self.blacklists = blacklists
        self.blacklists_indices = blacklists_indices
        self.replacement_indices = replacement_indices
        self.replacement_scaling_factors = replacement_scaling_factors

        self.n_features = len(features)
        self.feature_index_dict = dict([(feat, index) for index, feat in enumerate(features)])
        self.shape = (len(input_paths), *shape)
        self.data = None

    def get_feature_data(self, chrom, start, end, nan_as_zero=True, feature_indices=None):
        if not self.initialized:
            self.data = [pyBigWig.open(path) for path in self.input_paths]
            if self.blacklists is not None:
                self.blacklists = [tabix.open(blacklist) for blacklist in self.blacklists]
            self.initialized = True

        if feature_indices is None:
            feature_indices = np.arange(len(self.data))

        wigmat = np.zeros((len(feature_indices), end - start), dtype=np.float32)
        for i in feature_indices:
            try:
                wigmat[i, :] = self.data[i].values(chrom, start, end, numpy=True)
            except:
                print(chrom, start, end, self.input_paths[i], flush=True)
                raise

        if self.blacklists is not None:
            if self.replacement_indices is None:
                if self.blacklists_indices is not None:
                    for blacklist, blacklist_indices in zip(self.blacklists, self.blacklists_indices):
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[blacklist_indices, np.fmax(int(s) - start, 0): int(e) - start] = 0
                else:
                    for blacklist in self.blacklists:
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[:, np.fmax(int(s) - start, 0): int(e) - start] = 0
            else:
                for blacklist, blacklist_indices, replacement_indices, replacement_scaling_factor in zip(
                        self.blacklists, self.blacklists_indices, self.replacement_indices,
                        self.replacement_scaling_factors):
                    for _, s, e in blacklist.query(chrom, start, end):
                        wigmat[blacklist_indices, np.fmax(int(s) - start, 0): int(e) - start] = \
                            (wigmat[replacement_indices, np.fmax(int(s) - start, 0): int(e) - start]
                             * replacement_scaling_factor)

        if nan_as_zero:
            wigmat[np.isnan(wigmat)] = 0
        return wigmat


@register_dataset("promoter")
class TSSDatasetS(Dataset):
    def __init__(
            self, root, split,
            ref_file='Homo_sapiens.GRCh38.dna.primary_assembly.fa',
            ref_file_mmap='Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap',
            tsses_file='FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v4.tsv',
            fantom_files=('agg.plus.bw.bedgraph.bw', 'agg.minus.bw.bedgraph.bw'),
            fantom_blacklist_files=('fantom.blacklist8.plus.bed.gz', 'fantom.blacklist8.minus.bed.gz'),
            seqlength=1024, n_tsses=100000, rand_offset=0
    ):
        self.root = root
        self.split = split
        self.ref_file = ref_file
        self.ref_file_mmap = ref_file_mmap
        self.tsses_file = tsses_file
        self.fantom_files = fantom_files
        self.fantom_blacklist_files = fantom_blacklist_files
        self.seqlength = seqlength
        self.n_tsses = n_tsses
        self.rand_offset = rand_offset

        self.genome = MemmapGenome(
            root=root,
            input_path=ref_file,
            memmapfile=ref_file_mmap,
            blacklist_regions='hg38'
        )
        self.tfeature = GenomicSignalFeatures(
            [os.path.join(root, f) for f in fantom_files],
            ['cage_plus', 'cage_minus'],
            (2000,),
            [os.path.join(root, f) for f in fantom_blacklist_files],
        )

        self.tsses = pd.read_table(os.path.join(root, tsses_file), sep='\t')
        self.tsses = self.tsses.iloc[:n_tsses, :]
        self.chr_lens = self.genome.get_chr_lens()
        if split == "train":
            self.tsses = self.tsses.iloc[~np.isin(self.tsses['chr'].values, ['chr8', 'chr9', 'chr10'])]
        elif split == "valid":
            self.tsses = self.tsses.iloc[np.isin(self.tsses['chr'].values, ['chr10'])]
        elif split == "test":
            self.tsses = self.tsses.iloc[np.isin(self.tsses['chr'].values, ['chr8', 'chr9'])]
        else:
            raise ValueError

    def __len__(self):
        return self.tsses.shape[0]

    def __getitem__(self, tssi):
        chrm, pos, strand = (
            self.tsses['chr'].values[tssi],
            self.tsses['TSS'].values[tssi],
            self.tsses['strand'].values[tssi]
        )
        offset = 1 if strand == '-' else 0

        offset = offset + np.random.randint(-self.rand_offset, self.rand_offset + 1)
        seq = self.genome.get_encoding_from_coords(chrm, pos - int(self.seqlength / 2) + offset,
                                                   pos + int(self.seqlength / 2) + offset, strand)
        signal = self.tfeature.get_feature_data(chrm, pos - int(self.seqlength / 2) + offset,
                                                pos + int(self.seqlength / 2) + offset)
        if strand == '-':
            signal = signal[::-1, ::-1]
        return torch.FloatTensor(seq.copy()), torch.FloatTensor(signal.T[:, 0:1].copy())
