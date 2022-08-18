import praw

from requests import Session


# session = Session()
# session.proxies = {'https': '127.0.0.1:7890', 'http': '127.0.0.1:7890'}

# 配置账号密码
reddit = praw.Reddit(client_id='', # your client_id
                     client_secret='',  # your client_secret
                     user_agent='', # your user_agent
                     username='', # your username
                     password='', # your password
                     # requestor_kwargs={"session": session} # if you need to use a agent, you can use it
                     )


RawFile_names = ['demo.json']




RawData_path = '../Data/RawData'
DetailedData_path = '../Data/DetailedData'
DialogueData_path = '../Data/DialogueData'


# langs = ['en', 'zh', 'ru', 'ja', 'hu', 'de', 'ro', 'fr', 'ko', 'es', 'it', 'nl', 'ar', 'tr', 'hi', 'cs', 'lt', 'vi', 'fi']
langs = ['zh', 'ru', 'ja', 'hu', 'de', 'ro', 'fr', 'ko', 'es', 'it', 'nl', 'ar', 'tr', 'hi', 'cs', 'lt', 'vi', 'fi']


batch_size = 50




class Config(object):

    action: str = ''

    def __init__(self):
        self.reddit = reddit
        self.data_infos = {
            'RawFile_names': RawFile_names,
            'RawData_path': RawData_path,
            'DetailedData_path': DetailedData_path,
            'DialogueData_path': DialogueData_path,
            # 'Log_path': Log_path
        }
        self.langs = langs
        self.batch_size = batch_size

