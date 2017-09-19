from parser import DataReader
from nn import NN

epoch_count = 100
batch_size = 100
save_step = 100

if __name__ == '__main__':
    print('Creating network model...')
    nn = NN()
    print('Preparing data...')
    reader = DataReader('./train_data/')
    n = 0

    for _ in range(epoch_count):
        while reader.is_possible_batch(batch_size):
            x, y = reader.into_input(batch_size, verbose=1)
            x = x.reshape([x.shape[0] * x.shape[1], x.shape[2]])
            y = y.reshape([y.shape[0] * y.shape[1], y.shape[2]])
            nn.train(x, y, epoch=1)
            if n % save_step == 0:
                print('Saving the model...')
                nn.save()
            n += 1
        reader.reset()
