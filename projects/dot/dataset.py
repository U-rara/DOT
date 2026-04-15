import logging

import datasets

from verl.utils.dataset import RLHFDataset

logger = logging.getLogger(__name__)

answer_format = """
Please reason step by step, and put your final answer within \\boxed{{}}."""


class CustomRLHFDataset(RLHFDataset):
    """Custom dataset class to process Maxwell-Jia/AIME_2024, yentinglin/aime_2025 datasets."""

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset(parquet_file)["train"]
            data_source = "/".join(parquet_file.split("/")[-1:])
            if data_source.lower() in [
                "aime_2024",
                "aime_2025",
                "amc-23",
                "math-500",
                "deepscaler-preview-dataset",
            ]:
                print(f"process aime dataset: {data_source}")
                dataframe = dataframe.map(
                    self.map_fn_aime,
                    fn_kwargs={"data_source": data_source},
                    remove_columns=dataframe.column_names,
                )
            else:
                dataframe = dataframe.map(self.map_fn_normal, num_proc=32)
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

    def map_fn_aime(self, row: dict, *, data_source: str = None):
        if data_source.lower() == "aime_2024":
            problem, answer = row["Problem"], row["Answer"]
        elif data_source.lower() in ["aime_2025", "amc-23", "math-500", "deepscaler-preview-dataset"]:
            problem, answer = row["problem"], row["answer"]
        else:
            raise NotImplementedError

        prompt = problem + answer_format
        data = {
            "data_source": data_source.lower(),  # aime_2024, aime_2025, amc-23, math-500, deepscaler-preview-dataset
            "prompt": [{"role": "user", "content": prompt}],
            "reward_model": {"ground_truth": str(answer)},
            "agent_name": "tool_agent",
        }
        return data

    def map_fn_normal(self, row: dict):
        content = row["prompt"][0]["content"]
        row["prompt"][0]["content"] = content + answer_format
        row["agent_name"] = "tool_agent"
        return row
