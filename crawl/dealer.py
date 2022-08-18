import json
import os
import re

import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from langid.langid import LanguageIdentifier, model
from praw.exceptions import ClientException

from config import Config
from read_write import Writer


class Dealer(object):

    def __init__(self, config: Config):
        self.config = config
        self.action = config.action
        if self.action not in ['zip', 'generate']:
            raise Exception('please input correct action, zip or generate')

    def create_dealer(self):

        if self.action == 'zip':
            return Zipper(self.config)
        elif self.action == 'generate':
            return DialogueGenerator(self.config)


class Zipper(object):

    def __init__(self, config: Config):

        self.config = config
        self.batch_size = config.batch_size
        self.langs = config.langs
        self.log: dict = self.check_log()
        self.writer = None
        self.identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

    def check_log(self):

        log_path = f"{self.config.data_infos['RawData_path']}/log.json"
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    log = json.load(f)
            except json.decoder.JSONDecodeError:
                raise Exception('log.json error')
        else:
            log = {}

        return log

    def deal(self):
        for file in self.config.data_infos['RawFile_names']:
            self.writer = Writer(self.config, file)
            if file in self.log:
                if self.log[file] == 'done':
                    print(f'{file} has been processed.')
                    continue
                elif type(self.log[file]) == int:
                    done_lines = self.log[file]
                else:
                    raise Exception('log.json error')
            else:
                done_lines = 0

            count = done_lines
            file_path = self.config.data_infos['RawData_path'] + '/' + file
            if not os.path.exists(file_path):
                raise Exception(f'{file_path} is not exist')
            with open(file_path, 'r', encoding='utf-8') as f:

                while done_lines:
                    next(f)
                    done_lines -= 1


                detailed_data = []
                times = self.batch_size - 1
                for line in f:
                    line = json.loads(line)
                    detailed_data.append(self.zip(line))
                    if times:
                        times -= 1
                    else:
                        count += self.batch_size
                        self.write(detailed_data, count)
                        detailed_data = []
                        times = self.batch_size - 1

                self.write(detailed_data, 'done')

    def zip(self, line: dict):

        line['body'] = self.delete_emoji(line['body'])
        length = len(line['body'])
        if length < 5 or length > 1000:
            return None
        elif (lang := self.identifier.classify(line['body'])[0]) in self.langs:
            if self.identifier.classify(line['body'])[1] < 0.8:
                return None
            else:
                data = [lang, line['id'], line['author'], line['body'], line['link_id'], line['parent_id'],
                        line['no_follow'],
                        line['permalink'], line['subreddit']]
                return data

    def write(self, data, times):
        self.writer.write_detailed(data)
        self.writer.write_detailed_log(times)

    def delete_emoji(self, str):
        pat = "[(\U0001F000-\U0001F093)(\U0001F300-\U0001F64F)(\U0001F680-\U0001F6FF)(\u2600-\u2B55)(\U0001F57A-\U0001F991)\U0000FE0F]"
        return re.sub(pat, '', str)


class DialogueGenerator(object):

    def __init__(self, config: Config):
        self.config = config
        self.batch_size = config.batch_size
        self.langs = config.langs
        self.reddit = config.reddit
        self.log = None
        self.writer = None
        self.identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

    def check_log(self, raw_file_name):
        log_path = f"{self.config.data_infos['DetailedData_path']}/{raw_file_name}/log.json"
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    log = json.load(f)
            except json.decoder.JSONDecodeError:
                raise Exception('log.json error')
        else:
            log = {}

        return log

    def deal(self):
        for folder in self.config.data_infos['RawFile_names']:
            detailed_path = f"{self.config.data_infos['DetailedData_path']}/{folder}"
            if not os.path.exists(detailed_path):
                raise Exception(f"{detailed_path} is not exist")
            self.log = self.check_log(folder)
            files = os.listdir(detailed_path)
            files = [i for i in files if i[-4:] == '.csv']
            for file in files:
                if file[:2] not in self.config.langs:
                    continue
                if file in self.log:
                    if self.log[file] == 'done':
                        print(f'{file} has been done')
                        continue
                    elif type(self.log[file]) == int:
                        done_lines = self.log[file]
                    else:
                        raise Exception('log.json error')
                else:
                    done_lines = 0
                self.writer = Writer(self.config, folder, file)
                lang = file[:2]
                detect_lang = ['zh-cn', 'zh-tw'] if lang == 'zh' else [lang]

                count = 0
                file_path = f"{self.config.data_infos['DetailedData_path']}/{folder}/{file}"

                data = pd.read_csv(file_path, chunksize=self.batch_size)
                for block in data:
                    count += self.batch_size
                    dialogue_data = []

                    if count <= done_lines:
                        continue
                    else:
                        for line in block.itertuples():
                            dialogue_data.append(self.generate(line, detect_lang))
                        self.write(dialogue_data, count)
                self.write([None], 'done')

    def generate(self, line, lang):
        dialogue_data = None
        try:
            if line.parent_id[:2] == 't1':
                source_lang = detect(line.body)
                if source_lang in lang:
                    comment = self.reddit.comment(line.parent_id)
                    target_lang = detect(comment.body)
                    if target_lang in lang:
                        submission = comment.submission
                        dialogue_data = [lang[0][:2], submission.title, comment.body, line.body, line.link_id,
                                         comment.id, line.id]
        except (ClientException, LangDetectException):
            pass
        return dialogue_data

    def write(self, data, times):
        self.writer.write_dialogue(data)
        self.writer.write_dialogue_log(times)


if __name__ == '__main__':
    config = Config()
    config.action = 'generate'
    dealer = Dealer(config).create_dealer()
    # print(dealer.log)
    # print(type(dealer.log['demo.json']))

    dealer.deal()
