import os
import pickle
import types
import warnings
from typing import Callable

import numpy as np
import structlog
import torch.utils.data as torchdata

from . import heuristics
from .dataset import ActiveLearningDataset

log = structlog.get_logger(__name__)
pjoin = os.path.join


class ActiveLearningLoop:
    """Object that perform the active learning iteration.

    Args:
        dataset (ActiveLearningDataset): Dataset with some sample already labelled.
        get_probabilities (Function): Dataset -> **kwargs ->
                                        ndarray [n_samples, n_outputs, n_iterations].
        heuristic (Heuristic): Heuristic from baal.active.heuristics.
        query_size (int): Number of sample to label per step.
        max_sample (int): Limit the number of sample used (-1 is no limit).
        uncertainty_folder (Optional[str]): If provided, will store uncertainties on disk.
        ndata_to_label (int): DEPRECATED, please use `query_size`.
        **kwargs: Parameters forwarded to `get_probabilities`.
    """

    def __init__(
        self,
        dataset: ActiveLearningDataset,
        get_probabilities: Callable,
        heuristic: heuristics.AbstractHeuristic = heuristics.Random(),
        query_size: int = 1,
        max_sample=-1,
        uncertainty_folder=None,
        ndata_to_label=None,
        **kwargs,
    ) -> None:
        if ndata_to_label is not None:
            warnings.warn(
                "`ndata_to_label` is deprecated, please use `query_size`.", DeprecationWarning
            )
            query_size = ndata_to_label
        self.query_size = query_size
        self.get_probabilities = get_probabilities
        self.heuristic = heuristic
        self.dataset = dataset
        self.max_sample = max_sample
        self.uncertainty_folder = uncertainty_folder
        self.kwargs = kwargs

    def step(self, pool=None) -> bool:
        '''
        Perform an active learning step.

        Args
            pool (iterable): Optional dataset pool indices.
                             If not set, will use pool from the active set.

        Returns
            boolean, Flag indicating if we continue training.
        '''

        if pool is None:
            pool = self.dataset.pool
            if len(pool) > 0:
                # Limit number of samples
                if self.max_sample != -1 and self.max_sample < len(pool):
                    indices = np.random.choice(len(pool), self.max_sample, replace=False)
                    pool = torchdata.Subset(pool, indices)
                else:
                    indices = np.arange(len(pool))
        else:
            indices = None

        if len(pool) > 0:
            if isinstance(self.heuristic, heuristics.Random):
                probs = np.random.uniform(low=0, high=1, size=(len(pool), 1))
            else:
                probs = self.get_probabilities(pool, **self.kwargs)
            if probs is not None and (isinstance(probs, types.GeneratorType) or len(probs) > 0):
                to_label, uncertainty = self.heuristic.get_ranks(probs)
                if indices is not None:
                    to_label = indices[np.array(to_label)]
                if self.uncertainty_folder is not None:
                    # We save uncertainty in a file.
                    uncertainty_name = (
                        f"uncertainty_pool={len(pool)}" f"_labelled={len(self.dataset)}.pkl"
                    )
                    pickle.dump(
                        {
                            "indices": indices,
                            "uncertainty": uncertainty,
                            "dataset": self.dataset.state_dict(),
                        },
                        open(pjoin(self.uncertainty_folder, uncertainty_name), "wb"),
                    )
                if len(to_label) > 0:
                    self.dataset.label(to_label[: self.query_size])
                    return True
        return False


class ActiveLearningLoopHuman:
    """Object that perform the active learning iteration.

    Args:
        dataset (ActiveLearningDataset): Dataset with some sample already labelled.
        get_probabilities (Function): Dataset -> **kwargs ->
                                        ndarray [n_samples, n_outputs, n_iterations].
        heuristic (Heuristic): Heuristic from baal.active.heuristics.
        query_size (int): Number of sample to label per step.
        max_sample (int): Limit the number of sample used (-1 is no limit).
        uncertainty_folder (Optional[str]): If provided, will store uncertainties on disk.
        ndata_to_label (int): DEPRECATED, please use `query_size`.
        **kwargs: Parameters forwarded to `get_probabilities`.
    """

    def __init__(
        self,
        dataset: ActiveLearningDataset,
        get_probabilities: Callable,
        heuristic: heuristics.AbstractHeuristic = heuristics.Random(),
        query_size: int = 1,
        max_sample=-1,
        uncertainty_folder=None,
        ndata_to_label=None,
        human = None, 
        human_get_probabilities = None,
        **kwargs,
    ) -> None:
        if ndata_to_label is not None:
            warnings.warn(
                "`ndata_to_label` is deprecated, please use `query_size`.", DeprecationWarning
            )
            query_size = ndata_to_label
        self.query_size = query_size
        self.get_probabilities = get_probabilities
        self.heuristic = heuristic
        self.dataset = dataset
        self.max_sample = max_sample
        self.uncertainty_folder = uncertainty_folder
        self.kwargs = kwargs
        self.human = human
        self.human_get_probabilities = human_get_probabilities

    def step(self, pool=None) -> bool:
        '''
        Perform an active learning step.

        Args:
            pool (iterable): Optional dataset pool indices.
                             If not set, will use pool from the active set.
            Human: human labeler that has a get_probability method to reject labeling the samples

        Returns:
            boolean, Flag indicating if we continue training.
        '''

        if pool is None:
            pool = self.dataset.pool
            if len(pool) > 0:
                # Limit number of samples
                if self.max_sample != -1 and self.max_sample < len(pool):
                    indices = np.random.choice(len(pool), self.max_sample, replace=False)
                    pool = torchdata.Subset(pool, indices)
                else:
                    indices = np.arange(len(pool))
        else:
            indices = None

        if len(pool) > 0:
            if isinstance(self.heuristic, heuristics.Random):
                probs = np.random.uniform(low=0, high=1, size=(len(pool), 1))
            else:
                probs = self.get_probabilities(pool, **self.kwargs)
            if self.human_get_probabilities is not None:
                hbm_prob = self.human_get_probabilities(pool, **self.kwargs)
                all_probs = [probs, hbm_prob]

            if probs is not None and (isinstance(probs, types.GeneratorType) or len(probs) > 0):
                if self.human_get_probabilities is not None:
                    to_label, uncertainty = self.heuristic.get_ranks_human(all_probs)
                else:
                    to_label, uncertainty = self.heuristic.get_ranks(probs)
                if indices is not None:
                    to_label = indices[np.array(to_label)]
                if self.uncertainty_folder is not None:
                    # We save uncertainty in a file.
                    uncertainty_name = (
                        f"uncertainty_pool={len(pool)}" f"_labelled={len(self.dataset)}.pkl"
                    )
                    pickle.dump(
                        {
                            "indices": indices,
                            "uncertainty": uncertainty,
                            "dataset": self.dataset.state_dict(),
                        },
                        open(pjoin(self.uncertainty_folder, uncertainty_name), "wb"),
                    )
                # filter by human labeler
                indices_chosen = to_label[: self.query_size]
                final_indices = []
                for i in indices_chosen:
                    #x = self.dataset.get_raw(i)
                    x = self.dataset.pool[i]
                    x = x[0].numpy()
                    prob, label = self.human.hbm(x)
                    if label == 1:
                        final_indices.append(i)
                if len(final_indices) == 0:
                    return True, self.dataset._pool_to_oracle_index(to_label[: self.query_size])
                else:
                    if len(to_label) > 0:
                        tried_labeling = self.dataset._pool_to_oracle_index(to_label[: self.query_size])
                        self.dataset.label(final_indices)
                        return True, tried_labeling
        return False



class ActiveLearningLoopHuman_ColdStart:
    """Object that perform the active learning iteration.

    Args:
        dataset (ActiveLearningDataset): Dataset with some sample already labelled.
        get_probabilities (Function): Dataset -> **kwargs ->
                                        ndarray [n_samples, n_outputs, n_iterations].
        heuristic (Heuristic): Heuristic from baal.active.heuristics.
        query_size (int): Number of sample to label per step.
        max_sample (int): Limit the number of sample used (-1 is no limit).
        uncertainty_folder (Optional[str]): If provided, will store uncertainties on disk.
        ndata_to_label (int): DEPRECATED, please use `query_size`.
        **kwargs: Parameters forwarded to `get_probabilities`.
    """

    def __init__(
        self,
        dataset: ActiveLearningDataset,
        get_probabilities: Callable,
        heuristic: heuristics.AbstractHeuristic = heuristics.Random(),
        query_size: int = 1,
        max_sample=-1,
        uncertainty_folder=None,
        ndata_to_label=None,
        human = None, 
        human_get_probabilities = None,
        **kwargs,
    ) -> None:
        if ndata_to_label is not None:
            warnings.warn(
                "`ndata_to_label` is deprecated, please use `query_size`.", DeprecationWarning
            )
            query_size = ndata_to_label
        self.query_size = query_size
        self.get_probabilities = get_probabilities
        self.heuristic = heuristic
        self.dataset = dataset
        self.max_sample = max_sample
        self.uncertainty_folder = uncertainty_folder
        self.kwargs = kwargs
        self.human = human
        self.human_get_probabilities = human_get_probabilities

    def step(self, pool=None) -> bool:
        '''
        Perform an active learning step.

        Args:
            pool (iterable): Optional dataset pool indices.
                             If not set, will use pool from the active set.
            Human: human labeler that has a get_probability method to reject labeling the samples

        Returns:
            boolean, Flag indicating if we continue training.
        '''

        if pool is None:
            pool = self.dataset.pool
            if len(pool) > 0:
                # Limit number of samples
                if self.max_sample != -1 and self.max_sample < len(pool):
                    indices = np.random.choice(len(pool), self.max_sample, replace=False)
                    pool = torchdata.Subset(pool, indices)
                else:
                    indices = np.arange(len(pool))
        else:
            indices = None

        if len(pool) > 0:
            if isinstance(self.heuristic, heuristics.Random):
                probs = np.random.uniform(low=0, high=1, size=(len(pool), 1))
            else:
                probs = self.get_probabilities(pool, **self.kwargs)
            if self.human_get_probabilities is not None:
                hbm_prob = self.human_get_probabilities(pool, **self.kwargs)
                all_probs = [probs, hbm_prob]

            if probs is not None and (isinstance(probs, types.GeneratorType) or len(probs) > 0):
                if self.human_get_probabilities is not None:
                    #to_label, uncertainty = self.heuristic.get_ranks_human(all_probs)
                    to_label, uncertainty = self.heuristic.get_ranks_human(all_probs)
                else:
                    to_label, uncertainty = self.heuristic.get_ranks(probs)
                if indices is not None:
                    to_label = indices[np.array(to_label)]
                if self.uncertainty_folder is not None:
                    # We save uncertainty in a file.
                    uncertainty_name = (
                        f"uncertainty_pool={len(pool)}" f"_labelled={len(self.dataset)}.pkl"
                    )
                    pickle.dump(
                        {
                            "indices": indices,
                            "uncertainty": uncertainty,
                            "dataset": self.dataset.state_dict(),
                        },
                        open(pjoin(self.uncertainty_folder, uncertainty_name), "wb"),
                    )
                # filter by human labeler
                indices_chosen = to_label[: self.query_size]
                final_indices = []
                for i in indices_chosen:
                    #x = self.dataset.get_raw(i)
                    x = self.dataset.pool[i]
                    y = x[1]
                    x = x[0].numpy()
                    prob, label = self.human.hbm_init(x, y)
                    if label == 1:
                        final_indices.append(i)
                if len(final_indices) == 0:
                    return True, self.dataset._pool_to_oracle_index(to_label[: self.query_size])
                else:
                    if len(to_label) > 0:
                        tried_labeling = self.dataset._pool_to_oracle_index(to_label[: self.query_size])
                        self.dataset.label(final_indices)
                        return True, tried_labeling
        return False
    


class ActiveLearningLoopHuman_ColdStart_SELBALD:
    """Object that perform the active learning iteration.

    Args:
        dataset (ActiveLearningDataset): Dataset with some sample already labelled.
        get_probabilities (Function): Dataset -> **kwargs ->
                                        ndarray [n_samples, n_outputs, n_iterations].
        heuristic (Heuristic): Heuristic from baal.active.heuristics.
        query_size (int): Number of sample to label per step.
        max_sample (int): Limit the number of sample used (-1 is no limit).
        uncertainty_folder (Optional[str]): If provided, will store uncertainties on disk.
        ndata_to_label (int): DEPRECATED, please use `query_size`.
        **kwargs: Parameters forwarded to `get_probabilities`.
    """

    def __init__(
        self,
        dataset: ActiveLearningDataset,
        get_probabilities: Callable,
        heuristic: heuristics.AbstractHeuristic = heuristics.Random(),
        query_size: int = 1,
        max_sample=-1,
        uncertainty_folder=None,
        ndata_to_label=None,
        human = None, 
        human_get_probabilities = None,
        **kwargs,
    ) -> None:
        if ndata_to_label is not None:
            warnings.warn(
                "`ndata_to_label` is deprecated, please use `query_size`.", DeprecationWarning
            )
            query_size = ndata_to_label
        self.query_size = query_size
        self.get_probabilities = get_probabilities
        self.heuristic = heuristic
        self.dataset = dataset
        self.max_sample = max_sample
        self.uncertainty_folder = uncertainty_folder
        self.kwargs = kwargs
        self.human = human
        self.human_get_probabilities = human_get_probabilities

    def step(self, pool=None, oracle_ask_map = None) -> bool:
        '''
        Perform an active learning step.

        Args:
            pool (iterable): Optional dataset pool indices.
                             If not set, will use pool from the active set.
            Human: human labeler that has a get_probability method to reject labeling the samples

        Returns:
            boolean, Flag indicating if we continue training.
        '''

        if pool is None:
            pool = self.dataset.pool
            if len(pool) > 0:
                # Limit number of samples
                if self.max_sample != -1 and self.max_sample < len(pool):
                    indices = np.random.choice(len(pool), self.max_sample, replace=False)
                    pool = torchdata.Subset(pool, indices)
                else:
                    indices = np.arange(len(pool))
        else:
            indices = None

        if len(pool) > 0:
            if isinstance(self.heuristic, heuristics.Random):
                probs = np.random.uniform(low=0, high=1, size=(len(pool), 1))
            else:
                probs = self.get_probabilities(pool, **self.kwargs)
            if self.human_get_probabilities is not None:
                hbm_prob = self.human_get_probabilities(pool, **self.kwargs)
                all_probs = [probs, hbm_prob]

            if probs is not None and (isinstance(probs, types.GeneratorType) or len(probs) > 0):
                if self.human_get_probabilities is not None:
                    #to_label, uncertainty = self.heuristic.get_ranks_human(all_probs)
                    to_label, uncertainty = self.heuristic.get_ranks_human(all_probs)
                else:
                    to_label, uncertainty = self.heuristic.get_ranks(probs)
                # filter to_label by oracle_ask_map
                to_label_oracle_indices = self.dataset._pool_to_oracle_index(to_label)
                already_asked = np.where(oracle_ask_map==1)[0]
                to_label_already_asked = np.array([i in already_asked for i in to_label_oracle_indices])
                to_label = np.array(to_label)[~to_label_already_asked]

                if indices is not None:
                    to_label = indices[np.array(to_label)]
                if self.uncertainty_folder is not None:
                    # We save uncertainty in a file.
                    uncertainty_name = (
                        f"uncertainty_pool={len(pool)}" f"_labelled={len(self.dataset)}.pkl"
                    )
                    pickle.dump(
                        {
                            "indices": indices,
                            "uncertainty": uncertainty,
                            "dataset": self.dataset.state_dict(),
                        },
                        open(pjoin(self.uncertainty_folder, uncertainty_name), "wb"),
                    )
                # filter by human labeler
                    
                indices_chosen = to_label[: self.query_size]
                final_indices = []
                for i in indices_chosen:
                    #x = self.dataset.get_raw(i)
                    x = self.dataset.pool[i]
                    y = x[1]
                    x = x[0].numpy()
                    prob, label = self.human.hbm_init(x, y)
                    if label == 1:
                        final_indices.append(i)
                if len(final_indices) == 0:
                    # humans reject to label any samples
                    return True, self.dataset._pool_to_oracle_index(to_label[: self.query_size])
                else:
                    if len(to_label) > 0:
                        tried_labeling = self.dataset._pool_to_oracle_index(to_label[: self.query_size])
                        self.dataset.label(final_indices)
                        return True, tried_labeling
        return False