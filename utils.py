import os
import numpy as np
import h5py
import json
import torch
# from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import imageio
from PIL import Image

from eval_func.bleu.bleu import Bleu
from eval_func.rouge.rouge import Rouge
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor
# from eval_func.spice.spice import Spice


def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """
    print(karpathy_json_path)
    assert dataset in {'facad', 'coco', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    train_image_attr = []
    #train_character = []
    val_image_paths = []
    val_image_captions = []
    val_image_attr = []
    #val_character = []
    test_image_paths = []
    test_image_captions = []
    test_image_attr = []
    #test_character = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        attr = []
        char = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])  # [[0], [1], [2], [3], [4]]
        for a in img['sentences']:
            word_freq.update(a['attr'])
            attr.append(a['attr'])
            #char.append(a['passionate'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'facad' else os.path.join(
            image_folder, img['filename'])
        
        if img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)
            test_image_attr.append(attr)
            #test_character.append(char)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
            val_image_attr.append(attr)
            #val_character.append(char)
        elif img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
            train_image_attr.append(attr)
            #train_character.append(char)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    assert len(train_image_paths) == len(train_image_attr)
    assert len(val_image_paths) == len(val_image_attr)
    assert len(test_image_paths) == len(test_image_attr)

    #assert len(train_image_paths) == len(train_character)
    #assert len(val_image_paths) == len(val_character)
    #assert len(test_image_paths) == len(test_character)

    print("find {} training data, {} val data, {} test data".format(len(train_image_paths), len(val_image_paths), len(test_image_paths)))

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}  # word2id
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)
    print("{} words write into WORDMAP".format(len(word_map)))

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    #for impaths, imcaps, imattrs, chars, split in [(val_image_paths, val_image_captions, val_image_attr, val_character, 'VAL'),
    #                               (test_image_paths, test_image_captions, test_image_attr, test_character, 'TEST'),
    #                               (train_image_paths, train_image_captions, train_image_attr, train_character, 'TRAIN')]:
    for impaths, imcaps, imattrs, split in [(val_image_paths, val_image_captions, val_image_attr, 'VAL'),
                                   (test_image_paths, test_image_captions, test_image_attr, 'TEST'),
                                   (train_image_paths, train_image_captions, train_image_attr, 'TRAIN')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            enc_attr = []
            #enc_char = []
            caplens = []
            attrlens = []
            #charlens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)
                    attr = sample(imattrs[i], k=captions_per_image)
                    #char = sample(chars[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image
                assert len(attr) == captions_per_image
                #assert len(char) == captions_per_image

                # Read images
                img = imageio.imread(impaths[i])
                # img = imread(impaths[i])
                if len(img.shape) == 2:
                    # gray-scale
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)  # [256, 256, 1+1+1]
                img = np.array(Image.fromarray(img).resize((256, 256)))
                # img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

                for j, c in enumerate(attr):
                    # Encode captions
                    enc_a = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    a_len = len(c) + 2

                    enc_attr.append(enc_a)
                    attrlens.append(a_len)

                #for j, c in enumerate(char):
                    # Encode captions
                #    enc_ch = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                #        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                #    ch_len = len(c) + 2

                #    enc_char.append(enc_ch)
                #    charlens.append(ch_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)
            assert images.shape[0] * captions_per_image == len(enc_attr) == len(attrlens)
            #assert images.shape[0] * captions_per_image == len(enc_char) == len(charlens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)

            with open(os.path.join(output_folder, split + '_ATTR_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_attr, j)

            with open(os.path.join(output_folder, split + '_ATTRPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(attrlens, j)

            #with open(os.path.join(output_folder, split + '_CHAR_' + base_filename + '.json'), 'w') as j:
            #    json.dump(enc_char, j)

            #with open(os.path.join(output_folder, split + '_CHARLENS_' + base_filename + '.json'), 'w') as j:
            #    json.dump(charlens, j)


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    metrics, is_best, final_args):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'metrics': metrics,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer,
             'final_args': final_args}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def get_eval_score(references, hypotheses):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    hypo = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]]
    ref = [[' '.join(reft) for reft in reftmp] for reftmp in
           [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]

    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
        print("{} {}".format(method_i, score_i))
    score_dict = dict(zip(method, score))

    return score_dict


def convert2words(sequences, rev_word_map):
    for l1 in sequences:
        caption = ""
        for l2 in l1:
            caption += rev_word_map[l2]
            caption += " "
        print(caption)
