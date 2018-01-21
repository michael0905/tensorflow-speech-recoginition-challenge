from common import *
from dataset.audio_dataset import *

## sampling ###################################################################################

class TFRandomSampler(Sampler):
    def __init__(self, data, silence_probability=0.1, unknown_probability=0.1):
        self.data = data
        self.silence_probability = silence_probability
        self.unknown_probability = unknown_probability
        self.known_probability   = 1-silence_probability-unknown_probability

        known_num = 0
        for i in range(2,AUDIO_NUM_CLASSES):
            known_num += len(data.index_by_class[i])

        self.known_num   = known_num
        self.silence_num = int((self.known_num/self.known_probability)*self.silence_probability)
        self.unknown_num = int((self.known_num/self.known_probability)*self.unknown_probability)
        self.length = self.silence_num + self.unknown_num + self.known_num

    def __iter__(self):
        data = self.data
        l  = []
        if self.silence_num>0:#silence
            #empty (index is -1)
            silence_list = ([-1]+ data.index_by_class[0])*math.ceil(self.silence_num/(1+len(data.index_by_class[0])))
            random.shuffle(silence_list)
            silence_list = silence_list[:self.silence_num]
            l += silence_list

        if self.unknown_num>0:#unknown
            unknown_list = data.index_by_class[1]*math.ceil(self.unknown_num/len(data.index_by_class[1]))
            random.shuffle(unknown_list)
            unknown_list = unknown_list[:self.unknown_num]
            l +=  unknown_list

        for i in range(2,AUDIO_NUM_CLASSES):
            l +=  data.index_by_class[i]

        assert(len(l)==self.length)
        random.shuffle(l)
        return iter(l)


    def __len__(self):
        #print ('\tcalling Sampler:__len__')
        return self.length





class TFSequentialSampler(Sampler):
    def __init__(self, data, silence_probability=0.1, unknown_probability=0.1):
        self.data = data
        self.silence_probability = silence_probability
        self.unknown_probability = unknown_probability
        self.known_probability   = 1-silence_probability-unknown_probability

        known_num = 0
        for i in range(2,AUDIO_NUM_CLASSES):
            known_num += len(data.index_by_class[i])

        self.known_num   = known_num
        self.silence_num = int((self.known_num/self.known_probability)*self.silence_probability)
        self.unknown_num = int((self.known_num/self.known_probability)*self.unknown_probability)
        self.length = self.silence_num + self.unknown_num + self.known_num

        if self.unknown_num>0:
            unknown_list = data.index_by_class[1]*math.ceil(self.unknown_num/len(data.index_by_class[1]))
            random.shuffle(unknown_list)
            self.unknown_list = unknown_list[:self.unknown_num]
        else:
            self.unknown_list=[]


    def __iter__(self):
        data = self.data

        l  = []
        l += [-1]*self.silence_num  #silence (index is -1)
        l +=  self.unknown_list
        for i in range(2,AUDIO_NUM_CLASSES):
            l +=  data.index_by_class[i]
        assert(len(l)==self.length)
        return iter(l)


    def __len__(self):
        #print ('\tcalling Sampler:__len__')
        return self.length



## transform ###################################################################################

def tf_random_add_noise_transform(wave, noise_limit=0.2, u=0.5):

    if random.random() < u:
        num_noises = len(AUDIO_NOISES)
        noise = AUDIO_NOISES[np.random.choice(num_noises)]

        wave_length  = len(wave)
        noise_length = len(noise)
        t = np.random.randint(0, noise_length - wave_length - 1)
        noise = noise[t:t + wave_length]

        alpha = np.random.random() * noise_limit
        wave  = np.clip(alpha * noise + wave, -1, 1)

    return wave


def tf_random_time_shift_transform(wave, shift_limit=0.2, u=0.5):
    if random.random() < u:
        wave_length  = len(wave)
        shift_limit = shift_limit*wave_length
        shift = np.random.randint(-shift_limit, shift_limit)
        t0 = -min(0, shift)
        t1 =  max(0, shift)
        wave = np.pad(wave, (t0, t1), 'constant')
        wave = wave[:-t0] if t0 else wave[t1:]

    return wave


def tf_random_pad_transform(wave, length=AUDIO_LENGTH):

    if len(wave)<AUDIO_LENGTH:
        L = abs(len(wave)-AUDIO_LENGTH)
        start = np.random.choice(L)
        wave  = np.pad(wave, (start, L-start), 'constant')

    elif len(wave)>AUDIO_LENGTH:
        L = abs(len(wave)-AUDIO_LENGTH)
        start = np.random.choice(L)
        wave  = wave[start: start+AUDIO_LENGTH]

    return wave


def tf_fix_pad_transform(wave, length=AUDIO_LENGTH):
    # wave = np.pad(wave, (0, max(0, AUDIO_LENGTH - len(wave))), 'constant')
    # return wave

    if len(wave)<AUDIO_LENGTH:
        L = abs(len(wave)-AUDIO_LENGTH)
        start = L//2
        wave  = np.pad(wave, (start, L-start), 'constant')

    elif len(wave)>AUDIO_LENGTH:
        L = abs(len(wave)-AUDIO_LENGTH)
        start = L//2
        wave  = wave[start: start+AUDIO_LENGTH]

    return wave



def tf_random_scale_amplitude_transform(wave, scale_limit=0.1, u=0.5):
    if random.random() < u:
        scale = np.random.randint(-scale_limit, scale_limit)
        wave = scale*wave
    return wave
##  mfcc ,spectrogram ####################################################################

## https://www.youtube.com/watch?v=Gg4IHbiITd0
def tf_wave_to_mfcc(wave):

    spectrogram = librosa.feature.melspectrogram(wave, sr=AUDIO_SR, n_mels=40, hop_length=160, n_fft=480, fmin=20, fmax=4000)
    #spectrogram = librosa.power_to_db(spectrogram)
    idx = [spectrogram > 0]
    spectrogram[idx] = np.log(spectrogram[idx])

    dct_filters = librosa.filters.dct(n_filters=40, n_input=40)
    mfcc = [np.matmul(dct_filters, x) for x in np.split(spectrogram, spectrogram.shape[1], axis=1)]
    mfcc = np.hstack(mfcc)
    mfcc = mfcc.astype(np.float32)

    return mfcc


def tf_wave_to_melspectrogram(wave):
    spectrogram = librosa.feature.melspectrogram(wave, sr=AUDIO_SR, n_mels=40, hop_length=160, n_fft=480, fmin=20, fmax=4000)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)

    return spectrogram



def tf_wave_to_melspectrogram_mfcc(wave):

    spectrogram = librosa.feature.melspectrogram(wave, sr=AUDIO_SR, n_mels=40, hop_length=160, n_fft=480, fmin=5, fmax=4500)
    idx = [spectrogram > 0]
    spectrogram[idx] = np.log(spectrogram[idx])

    dct_filters = librosa.filters.dct(n_filters=40, n_input=40)
    mfcc = [np.matmul(dct_filters, x) for x in np.split(spectrogram, spectrogram.shape[1], axis=1)]
    mfcc = np.hstack(mfcc)
    mfcc = mfcc.astype(np.float32)

    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)

    all = np.concatenate((spectrogram[np.newaxis,:],mfcc[np.newaxis,:]))
    return all




##--------------
def tf_wave_to_melspectrogram1(wave):
    spectrogram = librosa.feature.melspectrogram(wave, sr=AUDIO_SR, n_mels=40, hop_length=160, n_fft=480, fmin=20, fmax=4000)
    idx = [spectrogram > 0]
    spectrogram[idx] = np.log(spectrogram[idx])
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


## check ## ----------------------------------------------------------
def run_check_tf_statics():
    dataset = AudioDataset( 'train_valid_6798', mode='train', )
    #sampler = SequentialSampler(dataset)
    sampler = RandomSampler(dataset)



    print('-----------------------------------')
    if 1:
        for n in iter(sampler):
            wave, label, index = dataset[n]

            if label==0: continue
            if label==1: continue
            print(label,AUDIO_NAMES[label])

            #mfcc = tf_wave_to_mfcc(wave)
            spectrogram = tf_wave_to_melspectrogram(wave)
            spectrogram1 = tf_wave_to_melspectrogram1(wave)
            # #spectrogram = spectrogram.reshape(-1)
            # #print(min(spectrogram),max(spectrogram))
            #
            # #plt.matshow(spectrogram[:,2:])
            # #plt.waitforbuttonpress(100)
            #
            # values = spectrogram[:,2:].reshape(-1)
            # min_value = -5#-10 #min(values) #-50
            # max_value = 10 #max(values) #-10
            # value = (np.clip(spectrogram,min_value,max_value)-min_value)/(max_value-min_value)*255
            # value = np.flipud(np.transpose(value))
            #
            # print(index,label,AUDIO_NAMES[label])
            # image_show('xxx',value,5)
            # cv2.waitKey(0)

            ax1 = plt.subplot(2,1,1)
            plt.title('spectrogram')
            librosa.display.specshow(spectrogram, sr=AUDIO_SR, x_axis='time', y_axis='mel')

            ax2 = plt.subplot(2,1,2)
            plt.title('spectrogram1')
            librosa.display.specshow(spectrogram1, sr=AUDIO_SR, x_axis='time', y_axis='mel')

            plt.tight_layout()
            plt.waitforbuttonpress(100)


    if 0:
        save_dir ='/root/share/project/kaggle/tensorflow/data/others/tf0'
        os.makedirs(save_dir,exist_ok=True)

        for i in range(2,AUDIO_NUM_CLASSES):
            index = dataset.index_by_class[i]
            for n in index:
                wave, label, index = dataset[n]

                spectrogram = tf_wave_to_spectrogram(wave)
                values = spectrogram[:,2:].reshape(-1)
                min_value = -5#-10 #min(values) #-50
                max_value = 10 #max(values) #-10
                value = (np.clip(spectrogram,min_value,max_value)-min_value)/(max_value-min_value)*255
                value = np.flipud(np.transpose(value))

                print(index,label,AUDIO_NAMES[label])
                image_show('xxx',value,5)
                cv2.waitKey(0)
                #print(n)

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_check_tf_statics()

    print( 'sucess!')
