from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
import sys, os
import warnings
import os.path as osp

import torch
from torch_geometric.data.dataset import (
    files_exist,
    to_list,
    _repr
)
from torch_geometric.io import fs
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData
from torch_geometric.data.dataset import Dataset, IndexType

class TemporalDataset:
    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        force_reload: bool = False,
    ) -> None:
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.log = log
        self.force_reload = force_reload

        self.data_dict = {'train': None,
                          'val': None,
                          'test': None}

        if self.has_download:
            self._download()

        if self.has_process:
            self._process()
        
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        r"""The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading.
        """
        raise NotImplementedError

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing.
        """
        raise NotImplementedError

    def download(self) -> None:
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError

    def process(self) -> None:
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError

    @property
    def raw_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading.
        """
        files = self.raw_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return [osp.join(self.raw_dir, f) for f in to_list(files)]

    @property
    def processed_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        processing.
        """
        files = self.processed_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return [osp.join(self.processed_dir, f) for f in to_list(files)]
    
    @property
    def has_download(self) -> bool:
        r"""Checks whether the dataset defines a :meth:`download` method."""
        return overrides_method(self.__class__, 'download')

    def _download(self):
        if files_exist(self.raw_paths):  # pragma: no cover
            return

        fs.makedirs(self.raw_dir, exist_ok=True)
        self.download()

    @property
    def has_process(self) -> bool:
        r"""Checks whether the dataset defines a :meth:`process` method."""
        return overrides_method(self.__class__, 'process')

    def _process(self):
        f = osp.join(self.processed_dir, 'pre_transform.pt')
        if osp.exists(f) and torch.load(f) != _repr(self.pre_transform):
            warnings.warn(
                "The `pre_transform` argument differs from the one used in "
                "the pre-processed version of this dataset. If you want to "
                "make use of another pre-processing technique, pass "
                "`force_reload=True` explicitly to reload the dataset.")

        f = osp.join(self.processed_dir, 'pre_filter.pt')
        if osp.exists(f) and torch.load(f) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in "
                "the pre-processed version of this dataset. If you want to "
                "make use of another pre-fitering technique, pass "
                "`force_reload=True` explicitly to reload the dataset.")

        if not self.force_reload and files_exist(self.processed_paths):
            return

        if self.log and 'pytest' not in sys.modules:
            print('Processing...', file=sys.stderr)

        fs.makedirs(self.processed_dir, exist_ok=True)
        self.process()

        path = osp.join(self.processed_dir, 'pre_transform.pt')
        fs.torch_save(_repr(self.pre_transform), path)
        path = osp.join(self.processed_dir, 'pre_filter.pt')
        fs.torch_save(_repr(self.pre_filter), path)

        if self.log and 'pytest' not in sys.modules:
            print('Done!', file=sys.stderr)
    
    @property
    def num_node_features(self) -> int:
        r"""Returns the number of features per node in the dataset."""
        return self.data_dict['train'].num_node_features

    @property
    def num_features(self) -> int:
        r"""Returns the number of features per node in the dataset.
        Alias for :py:attr:`~num_node_features`.
        """
        return self.num_node_features

    @property
    def num_edge_features(self) -> int:
        r"""Returns the number of features per edge in the dataset."""
        return self.data_dict['train'].num_edge_features

    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        return self.data_dict['train'].num_classes
    
    def __len__(self) -> int:
        r"""The number of examples in the dataset."""
        return 3

    def __iter__(self) -> Iterator[BaseData]:
        for i in range(len(self)):
            yield self.data_dict[['train', 'val', 'test'][i]]

    def __getitem__(self, key: Union[int, str]) -> BaseData:
        if isinstance(key, int):
            assert key in [0, 1, 2]
            return self.data_dict[['train', 'val', 'test'][key]]
        else:
            assert key in ['train', 'val', 'test']
            return self.data_dict[key]
    
    @property
    def data(self) -> Any:
        return self.data_dict['train']
    
def overrides_method(cls, method_name: str) -> bool:
    if method_name in cls.__dict__:
        return True

    out = False
    for base in cls.__bases__:
        if base != Dataset and base != TemporalDataset:
            out |= overrides_method(base, method_name)
    return out