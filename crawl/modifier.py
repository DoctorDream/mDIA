import os
import re
from typing import List

import pandas
import pandas as pd

from rule_pattern import *


class Modifier(object):
    def __init__(self, raw_data_path: str, save_path: str, langs: List[str], tokenizer=None):
        self.raw_data_path = raw_data_path
        self.save_path = save_path
        self.langs = langs
        self.tokenizer = tokenizer

    def doit(self):
        for lang in self.langs:
            path = f'{self.raw_data_path}/{lang}_dialogue_data.csv'
            if not os.path.exists(path):
                continue
            dialogue = pd.read_csv(path, lineterminator='\n')
            cleaned = self.through_cleanup(dialogue)
            save_path = f'{self.save_path}/{lang}_cleaned_dialogue_data.csv'
            self.write(save_path, cleaned)

    def through_cleanup(self, dialogue: pd.DataFrame):
        cleaned = pd.DataFrame(
            columns=['lang', 'title', 'source_body', 'target_body', 'link_id', 'source_id', 'target_id'])
        for index in dialogue.index:
            line: pd.Series = dialogue.loc[index]
            body = [line.source_body, line.target_body]
            if self.check_symbol(body):
                continue
            body = self.replace_symbol(body)
            if self.check_token_number(body):
                continue
            line['source_body'] = body[0]
            line['target_body'] = body[1]
            cleaned = cleaned.append(line, ignore_index=True)
        return cleaned

    def check_symbol(self, body: List[str]):
        try:
            if re.match(N_PATTERN, body[0]) or re.match(N_PATTERN, body[1]):
                return True
        except (TypeError) as e:
            print('error')
            return True
        return False

    def replace_symbol(self, body: List[str]):
        body = [re.compile(USER_PATTERN).sub('[USER]', body[0]),
                re.compile(USER_PATTERN).sub('[USER]', body[1])]
        body = [re.compile(SUBREDDIT_PATTERN).sub('[SUB]', body[0]),
                re.compile(SUBREDDIT_PATTERN).sub('[SUB]', body[1])]
        body = [re.compile(URL_PATTERN).sub('', body[0]),
                re.compile(URL_PATTERN).sub('', body[1])]
        body = [re.compile('\n').sub(' ', body[0]), re.compile('\n').sub(' ', body[1])]
        return body

    def check_token_number(self, body: List[str]):
        body = [self.tokenizer.tokenize(body[0]), self.tokenizer.tokenize(body[1])]
        if 4 < len(body[0]) < 80 and 4 < len(body[1]) < 80:
            return False
        else:
            return True

    def write(self, save_path, cleaned: pandas.DataFrame):
        if os.path.exists(self.save_path) is not True:
            os.mkdir(self.save_path)
        if os.path.exists(save_path):
            header = False
        else:
            header = True
        cleaned.to_csv(save_path, mode='a+', index=False, header=header, encoding='utf-8', line_terminator='\n')


if __name__ == '__main__':
    from transformers import AutoTokenizer

    model_name = 'google/mt5-base'

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    langs = [['de']]
    for month, lang in zip(['03'], langs):
        raw_data_path = f'../data/RC_2020-{month}'
        print(f"Dealing {raw_data_path[-10:]}")
        save_path = '../data/cleaned_data'
        m = Modifier(raw_data_path, save_path, lang, tokenizer)

        m.doit()

