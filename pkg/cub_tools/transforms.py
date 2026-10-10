import torchvision.transforms as transforms

# Default training and testing augmentations transforms
def makeDefaultTransforms(img_crop_size=224, img_resize=256):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(img_resize),
            transforms.CenterCrop(img_crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    return data_transforms

# More aggresive augmentation workflow with rotations

def makeAggresiveTransforms(
    img_crop_size=224,
    img_resize=256,
    transform_type='all',
    persp_distortion_scale=0.25,
    rotation_range=(-10.0, 10.0)
):
    test = transforms.Compose([
        transforms.Resize(img_resize),
        transforms.CenterCrop(img_crop_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    train_transforms = [
        transforms.RandomResizedCrop(img_crop_size),
        transforms.RandomHorizontalFlip()
    ]

    if transform_type in ['perspective', 'all']:
        train_transforms.append(
            transforms.RandomPerspective(
                distortion_scale=persp_distortion_scale,
                p=0.5,
                interpolation=transforms.InterpolationMode.BICUBIC
            )
        )

    if transform_type in ['rotation', 'all']:
        train_transforms.append(
            transforms.RandomRotation(
                degrees=rotation_range,
                interpolation=transforms.InterpolationMode.NEAREST,
                expand=False,
                center=None,
                fill=0
            )
        )

    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    train = transforms.Compose(train_transforms)

    return {'train': train, 'test': test}

# Just resize and central cropping for display
def resizeCropTransforms(img_crop_size=224, img_resize=256):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(img_resize),
            transforms.CenterCrop(img_crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(img_resize),
            transforms.CenterCrop(img_crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    return data_transforms