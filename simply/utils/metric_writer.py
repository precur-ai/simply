# Copyright 2024 The Simply Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Metric writers for experiments."""

import abc
from typing import Any
from absl import logging
import tensorboardX

_HAS_CLU = False


class BaseMetricWriter(abc.ABC):
  """Base class for metric writers."""

  @abc.abstractmethod
  def write_scalars(self, step: int, scalars: dict[str, Any]) -> None:
    pass

  @abc.abstractmethod
  def write_texts(self, step: int, texts: dict[str, str]) -> None:
    pass

  @abc.abstractmethod
  def flush(self) -> None:
    pass

  @abc.abstractmethod
  def close(self) -> None:
    pass


class TensorboardXMetricWriter(BaseMetricWriter):
  """Metric writer using tensorboardX."""

  def __init__(self, logdir: str, just_logging=False):
    self._writer = tensorboardX.SummaryWriter(logdir=logdir)
    self.just_logging = just_logging

  def write_scalars(self, step: int, scalars: dict[str, Any]) -> None:
    for key, value in scalars.items():
      if self.just_logging:
        logging.info('writing scalars: %s: %s', key, value)
      else:
        self._writer.add_scalar(key, value, step)

  def write_texts(self, step: int, texts: dict[str, str]) -> None:
    for key, text in texts.items():
      if self.just_logging:
        logging.info('writing texts: %s: %s', key, text)
      else:
        self._writer.add_text(key, text, step)

  def flush(self) -> None:
    self._writer.flush()

  def close(self) -> None:
    self._writer.close()


class WandbMetricWriter(BaseMetricWriter):
  """Metric writer using Weights & Biases."""

  def __init__(self, project='simply', name=None, dir=None, config=None):
    import wandb
    self._wandb = wandb
    self._run = wandb.init(
        project=project, name=name, dir=dir, config=config,
    )

  def write_scalars(self, step: int, scalars: dict[str, Any]) -> None:
    self._wandb.log(scalars, step=step)

  def write_texts(self, step: int, texts: dict[str, str]) -> None:
    self._wandb.log(
        {k: self._wandb.Html(v) for k, v in texts.items()},
        step=step,
    )

  def flush(self) -> None:
    pass

  def close(self) -> None:
    self._wandb.finish()


class MultiMetricWriter(BaseMetricWriter):
  """Delegates to multiple metric writers."""

  def __init__(self, writers):
    self._writers = writers

  def write_scalars(self, step: int, scalars: dict[str, Any]) -> None:
    for w in self._writers:
      w.write_scalars(step, scalars)

  def write_texts(self, step: int, texts: dict[str, str]) -> None:
    for w in self._writers:
      w.write_texts(step, texts)

  def flush(self) -> None:
    for w in self._writers:
      w.flush()

  def close(self) -> None:
    for w in self._writers:
      w.close()


def create_metric_writer(logdir: str, just_logging=False) -> BaseMetricWriter:
  """Creates a metric writer based on the environment."""
  return TensorboardXMetricWriter(logdir, just_logging=just_logging)


def create_wandb_metric_writer(
    project: str = 'simply',
    name: str | None = None,
    dir: str | None = None,
    config: dict[str, Any] | None = None,
) -> BaseMetricWriter:
  """Creates a W&B metric writer."""
  return WandbMetricWriter(
      project=project, name=name, dir=dir, config=config,
  )
