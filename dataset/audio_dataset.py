##  https://www.kaggle.com/c/tensorflow-speech-recognition-challenge
##  https://www.kaggle.com/davids1992/data-visualization-and-investigation
##  https://www.kaggle.com/davids1992/data-visualization-and-investigation

#from dataset.transform import *
#from dataset.sampler import *
from utility.file import *
#from utility.draw import *





AUDIO_DIR = 'C:/Users/45190/Desktop/speech-roc/input'
AUDIO_NUM_CLASSES = 12
AUDIO_NAMES =[ 'silence', 'unknown', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
assert(AUDIO_NUM_CLASSES==len(AUDIO_NAMES))

AUDIO_SR     = 16000 #sampling rate
AUDIO_LENGTH = 16000
sd.default.samplerate = AUDIO_SR

AUDIO_SILENCE = \
    librosa.core.load( AUDIO_DIR + '/silence.wav', sr=AUDIO_SR)[0]

AUDIO_NOISES=[]
for file in ['dude_miaowing.wav',  'pink_noise.wav',   'white_noise.wav', 'doing_the_dishes.wav',
             'exercise_bike.wav',  'running_tap.wav']:
    audio_file = AUDIO_DIR + '/train/audio/_background_noise_/' + file
    wave = librosa.core.load(audio_file, sr=AUDIO_SR)[0]
    AUDIO_NOISES.append(wave)






## <todo> draw waveform and spectrogram #################################
## https://www.kaggle.com/davids1992/data-visualization-and-investigation


def collate(batch):
    batch_size = len(batch)
    num = len(batch[0])
    indices = [batch[b][num-1]for b in range(batch_size)]
    tensors = torch.stack([batch[b][0]for b in range(batch_size)], 0)
    if batch[0][1] is None:
        labels = None
    else:
        labels = torch.from_numpy(np.array([batch[b][1]for b in range(batch_size)])).long()
    return [tensors,labels,indices]
	


# #data iterator ----------------------------------------------------------------
class AudioDataset(Dataset):

    def __init__(self, split, transform=None, mode='train'):
        super(AudioDataset, self).__init__()
        start = timer()

        self.split = split
        self.transform = transform
        self.mode = mode

        #label
        label_to_name = dict(zip(list(range(AUDIO_NUM_CLASSES)), AUDIO_NAMES))
        name_to_label = dict(zip( AUDIO_NAMES,list(range(AUDIO_NUM_CLASSES))))

        #read split
        lines = read_list_from_file(AUDIO_DIR + '/split/' + split, comment='#')
        num_classes = len(AUDIO_NAMES)
        ids=[]
        labels=[]
        index_by_class=[]
        for line in lines:
            ids.append(line)

        if mode in ['train']:
            for line in lines:
                name = line.split('/')[2]
                if name not in AUDIO_NAMES:
                    #continue
                    name='unknown'
                labels.append(name_to_label[name])

            for i in range(num_classes):
                index =np.where(np.array(labels)==i)[0]
                index_by_class.append(list(index))


        #save
        self.index_by_class = index_by_class
        self.ids    = ids
        self.labels = labels

        #print
        print('\ttime = %0.2f min'%((timer() - start) / 60))
        print('\tnum_ids = %d'%(len(self.ids)))
        print('\tnum_classes = %d'%(num_classes))
        if mode in ['train']:
            for i in range(num_classes):
                print('\t\t%2d   %16s = %5d (%0.3f)'%(i,AUDIO_NAMES[i],len(index_by_class[i]),len(index_by_class[i])/len(self.ids)))
        print('')


    def __getitem__(self, index):

        label = None

        if index==-1: #silence
            wave = AUDIO_SILENCE

            if self.mode in ['train']:
                label = 0
        else:
            audio_file = AUDIO_DIR + '/' + self.ids[index]
            wave = librosa.core.load(audio_file, sr=AUDIO_SR)[0]

            if self.mode in ['train']:
                label = self.labels[index]


        if self.transform is not None:
            return self.transform(wave, label, index)
        else:
            return wave, label, index


    def __len__(self):
        return len(self.ids)



## create samples ##
import scipy.io.wavfile

# http://codingmess.blogspot.sg/2010/02/how-to-make-wav-file-with-python.html
def run_make_silence_audio():
    audio_file='/root/share/project/kaggle/tensorflow/data/train/audio/silence/empty.wav'

    duration   = 1
    sample_rate = 16000 #22050

    wave = np.zeros(duration*sample_rate, np.float32)
    scipy.io.wavfile.write(audio_file,sample_rate,wave)

    zz=0


## check ## ----------------------------------------------------------
from dataset.audio_processing_tf import *

def run_check_dataset():

    def augment(wave,label,index):
        #wave = tf_add_noise_transform(audio)
        #wave = tf_time_shift_transform(audio)
        return wave,label,index

    dataset = AudioDataset(
        'pseudo-test1_71300', mode='train',
        #'train_valid_6798', mode='train',
        #'train_train_51088', mode='train',
        transform = augment,
    )
    #sampler = SequentialSampler(dataset)
    #sampler = ConstantSampler(dataset, [2523,4046,3987,30,1330]*5) #'left','right', 'on'
    sampler = TFRandomSampler(dataset)


    print('index, str(label), name, str(wave.shape)')
    print('-----------------------------------')
    for n in iter(sampler):
    #for n in range(10):
    #n=0
    #while 1:
        wave, label, index = dataset[n]
        #if label !=5: continue

        if label is not None:
            name = AUDIO_NAMES[label]
        else:
            name='None'

        print('%09d: %s,%-10s  : %s   '%(index, str(label), name, str(wave.shape)), end='')
        print(wave[0:4])

        if 1:
            #play_wav(AUDIO_DIR + '/' + dataset.folder + '/audio/' + dataset.ids[index])
            #audio = pitchshift(audio,1) d
            #wave = librosa.effects.pitch_shift(wave, sr=AUDIO_SR, n_steps=3, bins_per_octave=24)
            #wave = wave[::-1].copy()

            #pitch_shift
            #n_steps = np.random.random_integers(-3,3)
            #wave = librosa.effects.pitch_shift(wave, sr=AUDIO_SR, n_steps=n_steps)

            #time_stretch
            rate = np.random.uniform(0.75,1.75)
            wave = librosa.effects.time_stretch(wave,rate)
            sd.play(wave, blocking=True)

            #input('Press Enter to continue...\n')
        if 0:
            spectrogram = tf_wave_to_spectrogram(wave)
            min_value = -5#-10 #min(values) #-50
            max_value = 10 #max(values) #-10
            value = (np.clip(spectrogram,min_value,max_value)-min_value)/(max_value-min_value)*255
            value = np.flipud(np.transpose(value))

            print(index,label,AUDIO_NAMES[label])
            image_show('xxx',value,5)
            cv2.waitKey(0)




def run_check_sampler():

    dataset = AudioDataset(
        'train_valid_6798', 'train', mode='train',
    )
    #sampler = TFRandomSampler(dataset,0.1,0.1)
    #sampler = TFSequentialSampler(dataset,0.1,0.1)
    sampler = TFSequentialSampler(dataset,0.1,0)

    for n in range(5):
        print('--------------')
        for i in iter(sampler):
            print (i)




# main #################################################################
if __name__ == '__main__':


    print( '%s: calling main function ... ' % os.path.basename(__file__))
    #run_make_silence_audio()

    run_check_dataset()
    #run_check_sampler()


    print( 'sucess!')
