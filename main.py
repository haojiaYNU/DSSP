import hydra

from model import PassNet
from train import Train
from utils import set_seed


@hydra.main(version_base=None, config_path='E:/pythonProject/conf', config_name='config')
def main(config):
    #set_seed(config['seed'])

    t = Train(config)
    if config['method'] == 'sl-only':
        t.sl_only()
    elif config['method'] == 'semi-only':
        t.vime_self.encoder = PassNet()
        t.semi_sl()
    elif config['method'] == 'self-semi-sl':
        t.self_sl()
        t.semi_sl()
    t.test()


if __name__ == '__main__':
    main()
