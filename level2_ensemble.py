import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'3,2,1,0'
NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

from common import *
from utility.file import *

from net.rate import *
from net.loss import *

from dataset.audio_dataset import *

from dataset.audio_processing_tf import *


from net.model.simple1d_net import Simple1dNet as Net1
from net.model.se_resnet3 import SeResNet3 as Net2
from net.model.vggnet1 import VggNet1  as Net3
from FuseNet import FuseNet as Net

def pitch_transform(wave, u=0.5):
    if random.random() < u:
        n_steps = np.random.random_integers(-2,2)
        wave = librosa.effects.pitch_shift(wave, sr=AUDIO_SR, n_steps=n_steps)
    return wave

def time_stretch_transform(wave, u=0.5):
    if random.random() < u:
        rate = np.random.uniform(0.8,1.5)
        wave = librosa.effects.time_stretch(wave, rate)
    return wave

def ensemble_train_augment(wave,label,index):
    wave = pitch_transform(wave, u=0.5)
    wave = time_stretch_transform(wave, u=0.5)

    wave = tf_random_time_shift_transform(wave, shift_limit=0.2, u=0.5)
    wave = tf_random_add_noise_transform (wave, noise_limit=0.2, u=0.5)
    wave = tf_random_pad_transform(wave)

    spectrogram, mfcc = tf_wave_to_melspectrogram_mfcc(wave)
    wave = torch.from_numpy(wave[np.newaxis, :])
    spectrogram = torch.from_numpy(spectrogram[np.newaxis, :])
    mfcc = torch.from_numpy(mfcc[np.newaxis, :])
    return wave, spectrogram, mfcc, label, index

def heavy_test_random_add_noise_transform(wave, noise_limit=1, u=0.9):
    if random.random() < u:
        num_noises = len(AUDIO_NOISES)
        noise = AUDIO_NOISES[np.random.choice(num_noises)]
            
        wave_length = len(wave)
        noise_length = len(noise)
        if wave_length != noise_length:
            noise = np.tile(noise, wave_length//noise_length + 1)
            noise_length = len(noise)
            t = np.random.randint(0, noise_length - wave_length - 1)
            noise = noise[t:t + wave_length]
        else:
            pass
            
        alpha = np.random.random() * noise_limit
        wave = np.clip(alpha * noise + wave, -1, 1)
    return wave

net_1 = Net1(in_shape=(1,16000), num_classes=AUDIO_NUM_CLASSES, mode='features').cuda().eval()
net_2 = Net2(in_shape=(1, 40, 101), num_classes=AUDIO_NUM_CLASSES, mode='features').cuda().eval()
net_3 = Net3(in_shape=(1, 40, 101), num_classes=AUDIO_NUM_CLASSES, mode='features').cuda().eval()
CHECKPOINT_1 = 'C:/Users/45190/Desktop/results/rawwave/checkpoint/00020000_model.pth'
CHECKPOINT_2 = 'C:/Users/45190/Desktop/results/se-resnet3/checkpoint/00008000_model.pth'
CHECKPOINT_3 = 'C:/Users/45190/Desktop/results/vggnet-mfcc/checkpoint/00020000_model.pth'
net_1.load_state_dict(torch.load(CHECKPOINT_1, map_location=lambda storage, loc: storage))
net_2.load_state_dict(torch.load(CHECKPOINT_2, map_location=lambda storage, loc: storage))
net_3.load_state_dict(torch.load(CHECKPOINT_3, map_location=lambda storage, loc: storage))

def run_train():
    out_dir = 'C:/Users/45190/Desktop/results/ensemble'
    initial_checkpoint = None

    ## setup  -----------------
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')

    ## net ----------------------
    log.write('** net setting **\n')
    net = Net(num_classes=AUDIO_NUM_CLASSES).cuda()
    log.write('%s\n\n'%(type(net)))
    log.write('\n')

    ## optimiser ----------------------------------
    iter_accum  = 1
    batch_size  = 256  ##NUM_CUDA_DEVICES*512 #256//iter_accum #512 #2*288//iter_accum

    num_iters   = 1000  *1000
    iter_smooth = 20
    iter_log    = 500
    iter_valid  = 500
    iter_save   = [0, num_iters-1]\
                   + list(range(0,num_iters,1000))#1*1000

    #LR = StepLR([ (0, 0.01),  (200, 0.001),  (300, -1)])
    LR = None
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.01/iter_accum, momentum=0.9, weight_decay=0.0001)

    start_iter = 0
    start_epoch= 0.
    if initial_checkpoint is not None:
        checkpoint  = torch.load(initial_checkpoint.replace('_model.pth','_optimizer.pth'))
        start_iter  = checkpoint['iter' ]
        start_epoch = checkpoint['epoch']
        #optimizer.load_state_dict(checkpoint['optimizer'])


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    train_dataset = AudioDataset(
                                #'train_trainvalid_57886', mode='train',
                                'train_train_51088', mode='train',
                                transform = None)
    train_loader  = DataLoader(
                        train_dataset,
                        # sampler = TFRandomSampler(train_dataset,0.1,0.1),
                        sampler     = TFSequentialSampler(train_dataset),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 6,
                        pin_memory  = True,
                        collate_fn  = collate)

    valid_dataset = AudioDataset(
                                'train_test_6835', mode='train',
                                 transform = heavy_test_random_add_noise_transform)
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = TFSequentialSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 6,
                        pin_memory  = True,
                        collate_fn  = collate)

    log.write('\ttrain_dataset.split = %s\n'%(train_dataset.split))
    log.write('\tvalid_dataset.split = %s\n'%(valid_dataset.split))
    log.write('\tlen(train_dataset)  = %d\n'%(len(train_dataset)))
    log.write('\tlen(valid_dataset)  = %d\n'%(len(valid_dataset)))
    log.write('\tlen(train_loader)   = %d\n'%(len(train_loader)))
    log.write('\tlen(valid_loader)   = %d\n'%(len(valid_loader)))
    log.write('\tbatch_size  = %d\n'%(batch_size))
    log.write('\titer_accum  = %d\n'%(iter_accum))
    log.write('\tbatch_size*iter_accum  = %d\n'%(batch_size*iter_accum))
    log.write('\n')

    log.write('** start training here! **\n')
    log.write(' optimizer=%s\n'%str(optimizer) )
    log.write(' momentum=%f\n'% optimizer.param_groups[0]['momentum'])
    log.write(' LR=%s\n\n'%str(LR) )

    log.write(' waves_per_epoch = %d\n\n'%len(train_dataset))
    log.write(' rate   iter_k   epoch  num_m | valid_loss/acc | train_loss/acc | batch_loss/acc |  time    \n')
    log.write('--------------------------------------------------------------------------------------------\n')


    train_loss  = 0.0
    train_acc   = 0.0
    valid_loss  = 0.0
    valid_acc   = 0.0
    batch_loss  = 0.0
    batch_acc   = 0.0
    rate = 0

    start = timer()
    j = 0
    i = 0


    while  i<num_iters:  # loop over the dataset multiple times
        sum_train_loss = 0.0
        sum_train_acc  = 0.0
        sum = 0

        net.train()
        optimizer.zero_grad()
        for tensors, labels, indices in train_loader:
            wave, spectrogram, mfcc, labels, indices = ensemble_train_augment(wave,label,index)
            i = j/iter_accum + start_iter
            epoch = (i-start_iter)*batch_size*iter_accum/len(train_dataset) + start_epoch
            num_products = epoch*len(train_dataset)

            if i % iter_valid==0:
                net.eval()
                valid_loss, valid_acc = evaluate(net, valid_loader)
                net.train()

                print('\r',end='',flush=True)
                log.write('%0.4f  %5.1f k  %6.2f  %4.1f | %0.4f  %0.4f | %0.4f  %0.4f | %0.4f  %0.4f | %s \n' % \
                        (rate, i/1000, epoch, num_products/1000000, valid_loss, valid_acc, train_loss, train_acc, batch_loss, batch_acc, \
                         time_to_str((timer() - start)/60)))
                time.sleep(0.01)

            #if 1:
            if i in iter_save:
                torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(i))
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter'     : i,
                    'epoch'    : epoch,
                }, out_dir +'/checkpoint/%08d_optimizer.pth'%(i))



            # learning rate schduler -------------
            if LR is not None:
                lr = LR.get_rate(i)
                if lr<0 : break
                adjust_learning_rate(optimizer, lr/iter_accum)
            rate = get_learning_rate(optimizer)[0]*iter_accum


            # one iteration update  -------------
            labels = Variable(labels).cuda()
            features1 = net_1(Variable(wave, volatile=True).cuda()).detach()
            features2 = net_2(Variable(spectrogram, volatile=True).cuda()).detach()
            features3 = net_3(Variable(mfcc, volatile=True).cuda()).detach()

            features1 = Variable(features1.data)
            features2 = Variable(features2.data)
            features3 = Variable(features3.data)
            tensors = torch.cat((features1, features2, features3), 1)
            logits = data_parallel(net, tensors)
            probs = F.softmax(logits, dim=1)
            loss = F.cross_entropy(logits, labels)
            acc = top_accuracy(probs, labels, top_k=(1,))

            # accumulated update
            loss.backward()
            if j%iter_accum == 0:
                #torch.nn.utils.clip_grad_norm(net.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()


            # print statistics  ------------
            batch_acc  = acc[0][0]
            batch_loss = loss.data[0]
            sum_train_loss += batch_loss
            sum_train_acc  += batch_acc
            sum += 1
            if i%iter_smooth == 0:
                train_loss = sum_train_loss/sum
                train_acc  = sum_train_acc /sum
                sum_train_loss = 0.
                sum_train_acc  = 0.
                sum = 0

            print('\r%0.4f  %5.1f k  %6.2f  %4.1f | %0.4f  %0.4f | %0.4f  %0.4f | %0.4f  %0.4f | %s  %d,%d, %s' % \
                    (rate, i/1000, epoch, num_products/1000000, valid_loss, valid_acc, train_loss, train_acc, batch_loss, batch_acc,
                     time_to_str((timer() - start)/60) ,i,j, str(tensors.size())), end='',flush=True)
            j=j+1
        pass  #-- end of one data loader --
    pass #-- end of all iterations --


    if 1:
        torch.save(net.state_dict(),out_dir +'/checkpoint/%d_model.pth'%(i))
        torch.save({
            'optimizer': optimizer.state_dict(),
            'iter'     : i,
            'epoch'    : epoch,
        }, out_dir +'/checkpoint/%d_optimizer.pth'%(i))

    log.write('\n')

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()


    print('\nsucess!')
