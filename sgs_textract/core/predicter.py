import re
from typing import Union
from pathlib import Path

from tqdm import tqdm

from core.preprocessor import DataPreprocessor
from utils.logger import logger

import pandas as pd
import spacy


class ModelPredicter:
    def __init__(self, model: Union[str, Path], gpu_id: int):
        if gpu_id > -1:
            spacy.prefer_gpu(gpu_id=gpu_id)
        logger.info("Iniciando o modelo...")
        self.nlp = spacy.load(model)
        self.orgs_list = {}

    @classmethod
    def _polish_organizations(
        cls, org_name: str, organizations: list, *, max_variation: int = 5
    ):
        result = [
            text
            for text in organizations
            if text != org_name
            and re.search(rf"(^|\s+){re.escape(org_name)}($|\s+)", text, re.IGNORECASE)
        ]
        if len(result) == 0 or len(result) > max_variation:
            return None

        longest_name = max(result, key=len)
        return longest_name

    def predict(
        self,
        df: pd.DataFrame,
        source_column: str,
        output_column: str,
        max_variation: int,
    ):
        logger.info("Formatando o dados...")
        df.insert(
            df.columns.get_loc(source_column) + 1,
            source_column + "_TEMP",
            df[source_column].values,
        )
        df.insert(df.columns.get_loc(source_column) + 2, output_column, "")
        source_column = source_column + "_TEMP"
        df[source_column] = DataPreprocessor.format_column(df[source_column])
        df = df.sort_values(by=source_column)
        df = df.reset_index(drop=True)
        logger.info("Realizando predições...")

        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            text, response = str(row[source_column]), None

            last_row = i - 1
            if last_row != -1:
                last_text = str(df.at[last_row, source_column])
                last_response = str(df.at[last_row, output_column])
                if last_response and (last_text == text):
                    response = last_response

            if response is None:
                doc = self.nlp(text)

                if len(doc.ents) == 0:
                    continue
                elif len(doc.ents) == 1:
                    response = str(doc.ents[0])

            if response is not None:
                organization = self.orgs_list.get(response, {"rows": []})
                organization["rows"].append(i)

                self.orgs_list.update({response: organization})
                df.at[i, output_column] = response

        df[output_column] = DataPreprocessor.format_column(df[output_column])

        if max_variation is None or max_variation > 0:
            ORGS_LIST = []
            logger.info("Padronizando as saídas...")
            for org_name, org_data in tqdm(self.orgs_list.items()):
                new_org_name = self._polish_organizations(
                    org_name, self.orgs_list.keys(), max_variation=max_variation
                )
                if new_org_name is not None:
                    df.loc[org_data["rows"], output_column] = new_org_name

                ORGS_LIST.append(new_org_name if new_org_name is not None else org_name)

        # df[output_column] = [
        #     next((item for item in ORGS_LIST if item in text), existing)
        #     for text, existing in zip(df[source_column], df[output_column])
        # ]
        df[output_column] = [
            next((item for item in ORGS_LIST if item in text), existing)
            if pd.isna(existing) or existing == ""
            else existing
            for text, existing in zip(df[source_column], df[output_column])
        ]

        df = df.drop(columns=[source_column])
        return df
