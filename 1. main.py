from train_model import train
# from train_model_tta import train
SR = 'ldsrs2' # 'edsr', 'ldsrs2', 'sen2sr
batch_size = 4
device = 'cuda:1'
in_lr =  0.0001
num_epochs = 10
train_perc = 0.9
num_sample = 10000
begin_unsupervise_epoch = 5

if __name__ == '__main__':
    train(SR, device, batch_size, in_lr, num_epochs, train_perc, num_sample, begin_unsupervise_epoch)