import glob
from datasets import load_dataset, Dataset
from timeit import default_timer as timer

dataset = load_dataset(
    'parquet', 
    data_files={'train': glob.glob('data/world-heightmaps-256-parquet/*.parquet')},
    split='train',
    num_proc=8
)
dataset = dataset.with_format('jax')
#dataset = dataset.to_iterable_dataset().with_format('jax')
batch_size = 32
steps_per_epoch = len(dataset) // batch_size
print('steps_per_epoch:', steps_per_epoch)

dataset_iterator = dataset.iter(batch_size=batch_size)
batch = next(dataset_iterator)
print('batch shape:', batch['heightmap'].shape)

start = timer()
for batch in dataset_iterator:
    heightmap = batch['heightmap']
end = timer()

print('Time to load whole dataset:', end - start)
