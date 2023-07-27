import fid
from keras.preprocessing.image import ImageDataGenerator

img_size = (256, 256)
batch_size = 50
dtype = 'float32'
mmap_filename = '../data/temp/mmap_file'

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
    dataset1 = get_iterator('../../heightmaps/fid_test1', idg)
    dataset2 = get_iterator('../../heightmaps/fid_test2', idg)
    params, apply_fn = fid.get_inception_model()

    mmap_mu1, mmap_sigma1 = fid.compute_statistics_mmapped(
        params, 
        apply_fn,
        num_batches = len(dataset1),
        batch_size = batch_size,
        get_batch_fn = lambda: dataset1.next()[0],
        filename = mmap_filename,
        dtype = dtype
    )
    fid.save_statistics('../data/temp/mmap_stats.npz', mmap_mu1, mmap_sigma1)
    print(f'mmap_mu shape: {mmap_mu1.shape}, mmap_sigma shape: {mmap_sigma1.shape}')

    original_mu1, original_sigma1 = fid.compute_statistics(
        params, 
        apply_fn,
        num_batches = len(dataset1),
        get_batch_fn = lambda: dataset1.next()[0]
    )
    fid.save_statistics('../data/temp/original_stats.npz', original_mu1, original_sigma1)
    print(f'original_mu shape: {original_mu1.shape}, original_sigma shape: {original_sigma1.shape}')

    print(f'mmap_mu1 {mmap_mu1}')
    print(f'original_mu1 {original_mu1}')