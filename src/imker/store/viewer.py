import os
from pathlib import Path
import datetime
import pandas as pd
import yaml
from typing import Union
from .cacher import PickledBz2Cacher


class RepositoryViewer:
    def __init__(
        self,
        repo_dir: Union[str, Path],
    ):
        self.repo_dir = Path(repo_dir)

    def search_repo(self):
        savedir = Path(self.repo_dir)

        cachefiles = sorted([p for p in savedir.glob(f"**/*") if os.path.isfile(p.as_posix())])

        task_cache = [p for p in cachefiles if p.suffix != ".yml"]
        yml_cache = [p for p in cachefiles if p.suffix == ".yml"]

        df_task = self._format_df(task_cache).rename(columns={"path": "cachefile"})
        df_yml = self._format_df(yml_cache).rename(columns={"path": "config"})
        return df_task.merge(df_yml[["argId", "stateId", "config"]], on=["argId", "stateId"]).drop(
            ["argId", "stateId"], axis=1
        )

    def _format_df(self, cachefiles):
        dt = [
            [i]
            + [datetime.datetime.fromtimestamp(c.stat().st_ctime).isoformat()]
            + c.as_posix().split("/")[-6:-1]
            + [c.as_posix()]
            for i, c in enumerate(cachefiles)
        ]

        df = (
            pd.DataFrame(
                dt,
                columns=[
                    "task_id",
                    "lastUpdatedDate",
                    "repo",
                    "method",
                    "processor",
                    "argId",
                    "stateId",
                    "path",
                ],
            )
            .sort_values(["task_id", "lastUpdatedDate"], ascending=False)
            .reset_index(drop=True)
        )

        return df

    def load_cache(self, task_id: Union[str, int], cacher=None):
        if cacher is None:
            cacher = PickledBz2Cacher

        cacher = cacher()
        if isinstance(task_id, str):
            return cacher.load(task_id)
        elif isinstance(task_id, int):
            path = self.search_repo().pipe(lambda x: x[x["task_id"] == task_id]["cachefile"]).item()
            return cacher.load(path)
        else:
            raise ValueError("argument id must be int or str.")

    def load_config(self, task_id: Union[str, int]):
        if isinstance(task_id, str):
            with open(task_id, "r") as f:
                config = yaml.safe_load(f)
            return config

        elif isinstance(task_id, int):
            path = self.search_repo().pipe(lambda x: x[x["task_id"] == task_id]["config"]).item()
            with open(path, "r") as f:
                config = yaml.safe_load(f)
            return config
        else:
            raise ValueError("argument id must be int or str.")
