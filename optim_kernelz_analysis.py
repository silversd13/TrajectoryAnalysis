from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# try out different kernel sizes
conv_kernel_sizes = [3, 7, 11, 15, 21, 31, 41, 51]
max_kernel_sizes = [2, 4, 6, 8, 10]

losses = []
params = []

for l1_conv_ks in conv_kernel_sizes:
    for l1_max_ks in max_kernel_sizes:
        for l2_conv_ks in conv_kernel_sizes:
            for l2_max_ks in max_kernel_sizes:
                # skip if conv net has odd dimensions
                dim = (((((((1000 - (l1_conv_ks - 1)) - (l1_max_ks - 1) - 1) / 2 + 1) - (l2_conv_ks - 1)) - (l2_max_ks - 1) - 1) / 2 + 1))
                if (dim != int(dim)):
                    continue

                # set up tensorbardx for data vis of loss
                LOG_DIR = './runs/' + 'l1_conv_ks_' + str(l1_conv_ks) + '_l1_max_ks_' + str(l1_max_ks) + '_l2_conv_ks_' + str(l2_conv_ks) + '_l2_max_ks_' + str(l2_max_ks)
                print('\n\n' + 60*'*' + '\n' + LOG_DIR + '\n' + 60*'*' + '\n\n')

                event_acc = EventAccumulator(LOG_DIR + '/ConvNet_Adam_SL1_TotalLoss')
                event_acc.Reload()

                try:
                    print(event_acc.Tags())
                    loss = event_acc.Scalars('Loss')
                    losses.append(loss[-1][2])
                    params.append([l1_conv_ks, l1_max_ks, l2_conv_ks, l2_max_ks])
                except:
                    print()

# find minimum loss
min_loss_idx = losses.index(min(losses))
min_loss = losses[min_loss_idx]
min_params = params[min_loss_idx]

print('Min Loss: {}'.format(min_loss))
print('Optimal Params: \n - L1_conv_ks: {}\n - L1_max_ks: {}\n - L2_conv_ks: {}\n - L2_max_ks: {}'.format(
    min_params[0], min_params[1], min_params[2], min_params[3]))
