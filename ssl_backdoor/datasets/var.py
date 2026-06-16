from torchvision import transforms

# 数据集参数配置
dataset_params = {
    'cc3m': {
        'normalize': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        'image_size': 224
    },
    'cc3m_small': {
        'normalize': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        'image_size': 32
    },
    'imagenet': {
        'normalize': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        'image_size': 224,
        'num_classes': 1000,
    },
    'imagenet100': {
        'normalize': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        'image_size': 224,
        'num_classes': 100,
    },
    'imagenet20': {
        'normalize': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        'image_size': 224,
        'num_classes': 20,
    },
    'cifar10': {
        'normalize': transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]), 
        'image_size': 32,
        'num_classes': 10,
        'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    },
    'cifar100': {
        'normalize': transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]), 
        'image_size': 32,
        'num_classes': 100,
    },
    'stl10': {
        'normalize': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        'image_size': 96,
        'num_classes': 10,
    },
    'gtsrb': {
        'normalize': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        'image_size': 32,
        'num_classes': 43,
        'classes': ['Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 'Speed limit (70km/h)',
                    'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 'No passing',
                    'No passing for vehicles over 3.5 metric tons', 'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
                    'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left',
                    'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work',
                    'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
                    'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left',
                    'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons']
    },
    'caltech101': {
        'normalize': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        'image_size': 224,
        'num_classes': 100,
        'classes': ['face', 'leopard', 'motorbike', 'accordion', 'airplane', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon', 'car side', 'ceiling fan', 'cellphone', 'chair', 'chandelier', 'cougar body', 'cougar face', 'crab', 'crayfish', 'crocodile', 'crocodile head', 'cup', 'dalmatian', 'dollar bill', 'dolphin', 'dragonfly', 'electric guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'ferry', 'flamingo', 'flamingo head', 'garfield', 'gerenuk', 'gramophone', 'grand piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline skate', 'joshua tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 'scissors', 'scorpion', 'sea horse', 'snoopy', 'soccer ball', 'stapler', 'starfish', 'stegosaurus', 'stop sign', 'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water lilly', 'wheelchair', 'wild cat', 'windsor chair', 'wrench', 'yin yang']
    },
    'oxford_pets': {
        'normalize': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        'image_size': 224,
        'num_classes': 37,
        'classes': ['abyssinian', 'american bulldog', 'american pit bull terrier', 'basset hound', 'beagle', 'bengal', 'birman', 'bombay', 'boxer', 'british shorthair', 'chihuahua', 'egyptian mau', 'english cocker spaniel', 'english setter', 'german shorthaired', 'great pyrenees', 'havanese', 'japanese chin', 'keeshond', 'leonberger', 'maine coon', 'miniature pinscher', 'newfoundland', 'persian', 'pomeranian', 'pug', 'ragdoll', 'russian blue', 'saint bernard', 'samoyed', 'scottish terrier', 'shiba inu', 'siamese', 'sphynx', 'staffordshire bull terrier', 'wheaten terrier', 'yorkshire terrier']
    }
}