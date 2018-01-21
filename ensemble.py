import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'3,2,1,0'
NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

from common import *
from utility.file import *

from net.rate import *
from net.loss import *

from dataset.audio_dataset import *
from dataset.audio_processing_tf import *

def do_ensemble():
    test_num = 158538
    root = 'C:/Users/45190/Desktop/results'
    resnet = root + '/se-resnet3/submit/probs.uint8.memmap'
    rawwave = root + '/rawwave/submit/probs.uint8.memmap'
    rawwave2 = root + '/rawwave2/submit/probs.uint8.memmap' 
    vgg = root + '/vggnet/submit/probs.uint8.memmap'
    vgg_mfcc = root + '/vggnet-mfcc/submit/probs.uint8.memmap'
    resnet4 = root + '/se-resnet4/submit/probs.uint8.memmap'
    models = [resnet, rawwave, vgg, vgg_mfcc, resnet4, rawwave2]
    out_dir = 'C:/Users/45190/Desktop/results/ensemble'
    sample_submit = 'C:/Users/45190/Desktop/results/sample_submission/sample_submission.csv'
    csv_file = out_dir +'/submit/ensemble_submission_v3.csv'
    all_logits = np.zeros((len(models), test_num, AUDIO_NUM_CLASSES))

    for i,model in enumerate(models):
        logits = np.memmap(model, dtype='uint8', mode='r', shape=(test_num, AUDIO_NUM_CLASSES))
        all_logits[i] = logits

    probs = all_logits.mean(axis=0)
    labels  = np.argmax(probs, axis=1)
    labels  = [AUDIO_NAMES[l] for l in labels]
    submit = pd.read_csv(sample_submit)
    print(len(labels))
    print(submit.shape)
    submit['label'] = labels
    submit.to_csv(csv_file, index=False)

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    do_ensemble()


    print('\nsucess!')
