from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS

classes_exp = ('Whiteboard', 'Chessboard', 'Tripod', 'Vegetation Plant', 'Metal-Frame', 'Glass Window', 
               'Brick Wall', 'Miscellaneous', 'Concrete Ground', 'Metal-Vent', 'Metal Door Knob/Handle', 
               'Block Wall', 'Concrete Wall', 'Concrete Footing', 'Concrete Beam', 'Brick Ground', 
               'Glass door', 'Plastic Flyscreen', 'Plastic Label', 'Vegetation Ground', 'Soil', 
               'Metal Label', 'Metal Pipe', 'Metal Sheet', 'Metal Smooth Sheet', 'Woodchip Ground', 
               'Wood/Timber Smooth Door', 'Tiles Ground', 'Pebble-Concrete Beam', 'Pebble-Concrete Ground',
               'Pebble-Concrete Wall', 'Plastic Pipe', 'Metal Profiled Sheet', 'Metal-Pole', 
               'Wood/timber Vent', 'Wood/timber Wall', 'Wood Frame', 'Metal Smooth Door',
               'Plastic Vent', 'Concrete Window Sill', 'Metal Profiled Door')

palette_exp = [[199, 197, 189], [240, 212, 173], [2, 195, 145], [0, 122, 16], [80, 84, 195],
                [69, 174, 186], [212, 78, 78], [181, 42, 181], [139, 150, 228], [184, 117, 40],
                [167, 174, 171], [144, 142, 142], [222, 222, 222], [253, 149, 138], [160, 187, 213],
                [12, 229, 68], [82, 185, 203], [108, 152, 134], [189, 196, 26], [67, 145, 3],
                [232, 183, 100], [255, 0, 128], [32, 49, 228], [195, 127, 115], [255, 128, 64],
                [4, 108, 16], [13, 186, 180], [255, 128, 192], [34, 31, 153], [210, 184, 98],
                [37, 78, 32], [212, 204, 228], [172, 228, 99], [33, 98, 194], 
                [237, 128, 22], [128, 169, 150], [216, 182, 111], [188, 114, 203], [236, 233, 124],
                [187, 5, 74], [113, 134, 124]]



@DATASETS.register_module()
class HSIFacade(BaseSegDataset):
    """
    paper: https://arxiv.org/abs/2212.02749
    """

    METAINFO = dict(classes=classes_exp, palette=palette_exp)

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
