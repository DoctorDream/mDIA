import json
import os

import pandas as pd

from config import Config


class Writer(object):

    def __init__(self, config, raw_file_name, lang_file_name=None):

        self.config = config
        self.infos = config.data_infos
        self.raw_file_name = raw_file_name
        self.lang_file_name = lang_file_name
        self.RawData_path = self.infos['RawData_path']
        self.DetailedData_path = self.infos['DetailedData_path']
        self.DialogueData_path = self.infos['DialogueData_path']
        self.langs = self.config.langs
        self.dir_check()
        self.detailed_exist_tag, self.dialogue_exist_tag = self.file_check()
        self.detailed_log, self.dialogue_log = self.log_check()

    def dir_check(self):

        if os.path.exists(self.RawData_path) is not True:
            os.mkdir(self.RawData_path)
        if os.path.exists(self.DetailedData_path) is not True:
            os.mkdir(self.DetailedData_path)
        if os.path.exists(self.DialogueData_path) is not True:
            os.mkdir(self.DialogueData_path)
        if os.path.exists(f'{self.DetailedData_path}/{self.raw_file_name}') is not True:
            os.mkdir(f'{self.DetailedData_path}/{self.raw_file_name}')
        if os.path.exists(f'{self.DialogueData_path}/{self.raw_file_name}') is not True:
            os.mkdir(f'{self.DialogueData_path}/{self.raw_file_name}')

    def file_check(self):

        detailed, dialogue = {}, {}
        for lang in self.langs:
            detailed[lang] = True if os.path.exists(f'{self.DetailedData_path}/{self.raw_file_name}/{lang}_detailed_data.csv') else False
            dialogue[lang] = True if os.path.exists(f'{self.DialogueData_path}/{self.raw_file_name}/{lang}_dialogue_data.csv') else False

        return detailed, dialogue

    def log_check(self):

        detailed_log_path = f'{self.RawData_path}/log.json'
        dialogue_log_path = f'{self.DetailedData_path}/{self.raw_file_name}/log.json'

        if os.path.exists(detailed_log_path):
            with open(detailed_log_path, 'r', encoding='utf-8') as f:
                detailed_log = json.load(f)
        else:
            detailed_log = {}
        if os.path.exists(dialogue_log_path):
            with open(dialogue_log_path, 'r', encoding='utf-8') as f:
                dialogue_log = json.load(f)
        else:
            dialogue_log = {}
        return detailed_log, dialogue_log

    def write_detailed(self, data):
        data = [i for i in data if i]
        temp_data = {}
        for line in data:
            lang = line[0]
            if lang in temp_data:
                temp_data[lang].append(line[1:])
            else:
                temp_data[lang] = [line[1:]]


        for lang, data in temp_data.items():
            path = f'{self.DetailedData_path}/{self.raw_file_name}/{lang}_detailed_data.csv'
            data = pd.DataFrame(data, columns=['id', 'author', 'body', 'link_id', 'parent_id', 'no_follow', 'permalink',
                                               'subreddit'])
            if not self.detailed_exist_tag[lang]:
                head, self.detailed_exist_tag[lang] = True, True
            else:
                head = False

            data.to_csv(path, index=False, header=head, mode='a', encoding='utf-8')

    def write_detailed_log(self, times):
        log_path = f'{self.RawData_path}/log.json'
        self.detailed_log[self.raw_file_name] = times
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.detailed_log, f)
            print(f'{self.raw_file_name}: The first {times} data has been processed！')

    def write_dialogue(self, data):
        data = [i for i in data if i]
        if data:
            lang = data[0][0]

            path = f'{self.DialogueData_path}/{self.raw_file_name}/{lang}_dialogue_data.csv'
            data = pd.DataFrame(data, columns=['lang', 'title', 'source_body', 'target_body', 'link_id', 'source_id', 'target_id'])
            if not self.dialogue_exist_tag[lang]:
                head, self.dialogue_exist_tag[lang] = True, True
            else:
                head = False

            data.to_csv(path, index=False, header=head, mode='a', encoding='utf-8')

    def write_dialogue_log(self, times):
        log_path = f'{self.DetailedData_path}/{self.raw_file_name}/log.json'
        self.dialogue_log[self.lang_file_name] = times
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.dialogue_log, f)
            print(f'{self.lang_file_name}: The first {times} data has been processed！')




if __name__ == '__main__':
    config = Config()
    data_infos = config.data_infos
    langs = config.langs
    writer = Writer(data_infos, langs)
