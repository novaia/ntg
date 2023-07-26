import fid
from keras.preprocessing.image import ImageDataGenerator

img_size = (256, 256)
batch_size = 50

def get_iterator(path, data_generator):
    iterator = data_generator.flow_from_directory(
        path,
        target_size = img_size,
        batch_size = batch_size,
        color_mode = 'rgb',
        classes = ['']
    )
    return iterator

def preprocessing_function(image):
    image = image.astype(float) / 255
    return image

if __name__ == '__main__':
    idg = ImageDataGenerator(preprocessing_function = preprocessing_function)
    dataset1 = get_iterator('../../fid_test1', idg)
    dataset2 = get_iterator('../../fid_test2', idg)
    params, apply_fn = fid.get_inception_model()

    mmap_mu1, mmap_sigma1 = fid.compute_statistics_mmapped(
        params, 
        apply_fn,
        num_batches = len(dataset2),
        get_batch_fn = lambda: dataset2.next()[0],
        filename = 'mmap_file',
        dtype = 'float32'
    )

    mmap_mu2, mmap_sigma2 = fid.compute_statistics_mmapped(
        params, 
        apply_fn,
        num_batches = len(dataset2),
        get_batch_fn = lambda: dataset2.next()[0],
        filename = 'mmap_file',
        dtype = 'float32'
    )
    print(f'mmap_mu shape: {mmap_mu1.shape}, mmap_sigma shape: {mmap_sigma1.shape}')

    original_mu1, original_sigma1 = fid.compute_statistics(
        params, 
        apply_fn,
        num_batches = len(dataset1),
        get_batch_fn = lambda: dataset1.next()[0]
    )
    original_mu2, original_sigma2 = fid.compute_statistics(
        params, 
        apply_fn,
        num_batches = len(dataset2),
        get_batch_fn = lambda: dataset2.next()[0]
    )
    print(f'original_mu shape: {mmap_mu1.shape}, original_sigma shape: {mmap_sigma1.shape}')

    mmap_fid = fid.calculate_frechet_distance(
        mmap_mu1, mmap_sigma1, mmap_mu2, mmap_sigma2
    )
    print(f'mmap_fid: {mmap_fid}')

    original_fid = fid.compute_frechet_distance(
        original_mu1, original_mu2, original_sigma1, original_sigma2
    )
    print(f'original_fid: {original_fid}')