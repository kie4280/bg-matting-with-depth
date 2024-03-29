"""
This file records the directory paths to the different datasets.
You will need to configure it for training the model.

All datasets follows the following format, where fgr and pha points to directory that contains jpg or png.
Inside the directory could be any nested formats, but fgr and pha structure must match. You can add your own
dataset to the list as long as it follows the format. 'fgr' should point to foreground images with RGB channels,
'pha' should point to alpha images with only 1 grey channel.
{
    'YOUR_DATASET': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        }
    }
}
"""

DATA_PATH = {
    'videomatte240k': {
        'train': {
            'fgr': '/eva_data/kie/data/training/VideoMatte240K_JPEG_HD/train/fgr',
            'pha': '/eva_data/kie/data/training/VideoMatte240K_JPEG_HD/train/pha'
        },
        'valid': {
            'fgr': '/eva_data/kie/data/training/foreground',
            'pha': '/eva_data/kie/data/training/alpha'
        }
    },
    'photomatte85': {
        'train': {
            'fgr': '/eva_data/kie/data/training/foreground',
            'pha': '/eva_data/kie/data/training/alpha'
        },
        'valid': {
            'fgr': '/eva_data/kie/data/validate/foreground',
            'pha': '/eva_data/kie/data/validate/alpha'
        }
    },
    'depth': {
        'train': {
            'depth': '/eva_data/kie/data/training/videos/output'
        },
        'valid': {
            'depth': '/eva_data/kie/data/training/videos/output'
        }
    },
    'distinction': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR'
        },
    },
    'adobe': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR'
        },
    },
    'backgrounds': {
        'train': '/eva_data/kie/data/training/videos/output',
        'valid': '/eva_data/kie/data/validate/background'
    },
}
