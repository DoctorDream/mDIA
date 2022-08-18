from config import Config
from dealer import Dealer
import sys

class Solution(object):

    def __init__(self, config: Config):
        self.config = config
        self.batch_size = config.batch_size
        self.dealer = Dealer(config).create_dealer()
        self.doit()

    def doit(self):
        self.dealer.deal()


def main():

    config = Config()
    if len(sys.argv) == 1:
        pass
    elif len(sys.argv) > 1:
        config.action = sys.argv[1]

    if len(sys.argv) > 2:
        config.data_infos['RawFile_names'] = [sys.argv[2]]
    else:
        raise Exception("Please input correct action and file name")
    # config.action = 'zip'
    Solution(config)


if __name__ == '__main__':
    main()
