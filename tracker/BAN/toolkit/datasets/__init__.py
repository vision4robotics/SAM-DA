from .uav import UAVDataset
from .UAVDark70 import UAVDark70Dataset
from .UAVDark135 import UAVDark135Dataset
from .DarkTrack2021 import DarkTrack2021Dataset
from .nat import NATDataset
from .nat_l import NAT_LDataset
from .nut import NUTDataset
from .nut_l import NUT_LDataset
from .nut_l_t import NUT_L_tDataset
from .nut_l_s import NUT_L_sDataset
class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name: dataset name 'UAVDark70', 'UAV', 'NAT', 'NAT'
            dataset_root: dataset root
            load_img: wether to load image
        Return:
            dataset
        """
        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        if 'UAVDark70' == name:
            dataset = UAVDark70Dataset(**kwargs)
        elif 'UAVDark135' == name:
            dataset = UAVDark135Dataset(**kwargs)
        elif 'DarkTrack' in name:
            dataset = DarkTrack2021Dataset(**kwargs)
        elif 'UAV' in name:
            dataset = UAVDataset(**kwargs)
        elif 'NAT' == name:
            dataset = NATDataset(**kwargs)
        elif 'NAT_L' == name:
            dataset = NAT_LDataset(**kwargs)
        elif 'NUT' == name:
            dataset = NUTDataset(**kwargs)
        elif 'NUT_L' == name:
            dataset = NUT_LDataset(**kwargs)
        elif 'NUT_L_target' == name:
            dataset = NUT_L_tDataset(**kwargs)
        elif 'NUT_L_source' == name:
            dataset = NUT_L_sDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

