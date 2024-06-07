#!/usr/bin/env python3

import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import *
from ACC_adaptive import *
from datasets_attrlen import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import argparse
import codecs
import numpy as np
import random
import torch.nn.functional as F

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def train(args, train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens, attrs, attrlens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        attrs = attrs.to(device)
        attrlens = attrlens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        # imgs: [batch_size, 14, 14, 2048]
        # caps: [batch_size, 52]
        # caplens: [batch_size, 1]
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens, attrs, attrlens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        # print(scores.size())
        # print(targets.size())

        # Calculate loss
        loss = criterion(scores, targets)

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if args.grad_clip is not None:
            clip_gradient(decoder_optimizer, args.grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, args.grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()
        if i % args.print_freq == 0:
            print("Epoch: {}/{} step: {}/{} Loss: {} AVG_Loss: {} Top-5 Accuracy: {} Batch_time: {}s".format(epoch+1, args.epochs, i+1, len(train_loader), losses.val, losses.avg, top5accs.val, batch_time.val))


def validate(args, val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: score_dict {'Bleu_1': 0., 'Bleu_2': 0., 'Bleu_3': 0., 'Bleu_4': 0., 'METEOR': 0., 'ROUGE_L': 0., 'CIDEr': 1.}
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    beam_size = 5
    Caption_End = False

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    with torch.no_grad():
        for i, (image, caps, caplens, allcaps, attrs, attrlens) in enumerate(val_loader):
            k = beam_size
            # Move to GPU device, if available
            image = image.to(device)  # [1, 3, 256, 256]
            attrs = attrs.to(device) #1, 32
            attrlens = attrlens.to(attrlens)

            # Encode
            encoder_out = encoder(image)  # [1, enc_image_size=14, enc_image_size=14, encoder_dim=2048]
            enc_image_size = encoder_out.size(1)
            encoder_dim = encoder_out.size(-1)
            # We'll treat the problem as having a batch size of k, where k is beam_size
            encoder_out = encoder_out.expand(k, enc_image_size, enc_image_size, encoder_dim)  # [k, enc_image_size, enc_image_size, encoder_dim]

            attrs = attrs.expand(k, attrs.shape[1])
            # Tensor to store top k previous words at each step; now they're just <start>
            # Important: [1, 52] (eg: [[<start> <start> <start> ...]]) will not work, since it contains the position encoding
            k_prev_words = torch.LongTensor([[word_map['<start>']]*27] * k).to(device)  # (k, 52)
            # Tensor to store top k sequences; now they're just <start>
            seqs = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1).to(device)
            # Lists to store completed sequences and scores
            complete_seqs = []
            complete_seqs_scores = []
            step = 1

            # Start decoding
            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:
                # print("steps {} k_prev_words: {}".format(step, k_prev_words))
                # cap_len = torch.LongTensor([52]).repeat(k, 1).to(device) may cause different sorted results on GPU/CPU in transformer.py
                cap_len = torch.LongTensor([27]).repeat(k, 1)  # [s, 1]
                attrlens = torch.LongTensor([7]).repeat(k, 1) 
                scores, _, _, _, _ = decoder(encoder_out, k_prev_words, cap_len, attrs, attrlens)
                scores = scores[:, step-1, :].squeeze(1)  # [s, 1, vocab_size] -> [s, vocab_size]
                scores = F.log_softmax(scores, dim=1)
                # top_k_scores: [s, 1]
                scores = top_k_scores.expand_as(scores) + scores  # [s, vocab_size]
                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words // vocab_size  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)

                # Add new words to sequences
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
                # Set aside complete sequences
                if len(complete_inds) > 0:
                    Caption_End = True
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly
                # Proceed with incomplete sequences
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                attrs = attrs[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                # Important: this will not work, since decoder has self-attention
                # k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1).repeat(k, 52)
                k_prev_words = k_prev_words[incomplete_inds]
                k_prev_words[:, :step+1] = seqs  # [s, 52]
                # k_prev_words[:, step] = next_word_inds[incomplete_inds]  # [s, 52]
                # Break if things have been going on too long
                if step > 27:
                    break
                step += 1

            # choose the caption which has the best_score.
            assert Caption_End
            indices = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[indices]
            # References
            img_caps = allcaps[0].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            references.append(img_captions)
            # Hypotheses
            # tmp_hyp = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
            hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
            assert len(references) == len(hypotheses)
            if i % (args.print_freq * 10) == 0:
                print(i)

    # Calculate BLEU1~4, METEOR, ROUGE_L, CIDEr scores
    metrics = get_eval_score(references, hypotheses)

    print("BLEU-1 {} BLEU2 {} BLEU3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format
          (metrics["Bleu_1"],  metrics["Bleu_2"],  metrics["Bleu_3"],  metrics["Bleu_4"],
           metrics["METEOR"], metrics["ROUGE_L"], metrics["CIDEr"]))

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image_Captioning')
    # Data parameters
    
    parser.add_argument('--data_folder', default="/mnt/data/facad/dataset3a/dataset/",
                        help='folder with data files saved by create_input_files.py.')
    parser.add_argument('--data_name', default="facad_1_cap_per_img_5_min_word_freq",
                        help='base name shared by data files.')
    # Model parameters
    parser.add_argument('--emb_dim', type=int, default=512, help='dimension of word embeddings.')
    parser.add_argument('--att_emb_dim', type=int, default=300, help='dimension of word embeddings.')
    parser.add_argument('--attention_dim', type=int, default=512, help='dimension of attention linear layers.')
    parser.add_argument('--decoder_dim', type=int, default=512, help='dimension of decoder RNN.')
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--dropout_att', type=float, default=0.5, help='dropout_att')
    parser.add_argument('--decoder_mode', default="transformer", help='which model does decoder use?')  # lstm or transformer
    parser.add_argument('--attention_method', default="ByPixel", help='which attention method to use?')  # ByPixel or ByChannel
    parser.add_argument('--attr_encoder_layers', type=int, default=1, help='the number of layers of encoder in Transformer.')
    parser.add_argument('--encoder_layers', type=int, default=1, help='the number of layers of encoder in Transformer.')
    parser.add_argument('--decoder_layers', type=int, default=6, help='the number of layers of decoder in Transformer.')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=45,
                        help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--stop_criteria', type=int, default=5, help='training stop if epochs_since_improvement == stop_criteria')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches.')
    parser.add_argument('--workers', type=int, default=1, help='for data-loading; right now, only 1 works with h5pys.')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at an absolute value of.')
    parser.add_argument('--alpha_c', type=float, default=1.,
                        help='regularization parameter for doubly stochastic attention, as in the paper.')
    parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='whether fine-tune encoder or not')
    parser.add_argument('--fine_tune_embedding', type=bool, default=False, help='whether fine-tune word embeddings or not')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint, None if none.')
    #parser.add_argument('--checkpoint', default="./BEST_checkpoint_facad_1_cap_per_img_5_min_word_freq.pth.tar", help='path to checkpoint, None if none.')
    parser.add_argument('--embedding_path', default=None, help='path to pre-trained word Embedding.')
    #parser.add_argument('--embedding_path', default="./dataset/Glove/glove.6B.300d.txt", help='path to pre-trained word Embedding.')
    args = parser.parse_args()

    # load checkpoint, these parameters can't be modified
    final_args = {"emb_dim": args.emb_dim,
                 "att_emb_dim" : args.att_emb_dim,
                 "attention_dim": args.attention_dim,
                 "decoder_dim": args.decoder_dim,
                 "n_heads": args.n_heads,
                 "dropout": args.dropout,
                 "dropout_att": args.dropout_att,
                 "decoder_mode": args.decoder_mode,
                 "attention_method": args.attention_method,
                 "encoder_layers": args.encoder_layers,
                 "attr_encoder_layers": args.attr_encoder_layers,
                 "decoder_layers": args.decoder_layers}
    print(final_args)
    start_epoch = 0
    best_cider = 0.  # BLEU-4 score right now
    epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    torch.cuda.set_device(0)
    print(torch.cuda.current_device())
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
    print(device)

    # Read word map
    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    print(word_map_file)
    print(len(word_map))
    vocab_size=len(word_map)

    word_map_file_attr = os.path.join(args.data_folder, 'WORDMAPATTR_' + args.data_name + '.json')
    with open(word_map_file_attr, 'r') as j:
        word_map_attr = json.load(j)
    print(word_map_file_attr)
    print(len(word_map_attr))

    # Initialize / load checkpoint
    if args.checkpoint is None:
        encoder = CNN_Encoder(attention_method=args.attention_method)
        encoder.fine_tune(args.fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=args.encoder_lr) if args.fine_tune_encoder else None

        if args.decoder_mode == "lstm":
            decoder = DecoderWithAttention(attention_dim=args.attention_dim,
                                           embed_dim=args.emb_dim,
                                           decoder_dim=args.decoder_dim,
                                           vocab_size=len(word_map),
                                           dropout=args.dropout)
        elif args.decoder_mode == "transformer":
            decoder = Transformer(vocab_size=len(word_map), vocab_size_attr=len(word_map_attr) , embed_dim=args.emb_dim, encoder_layers=args.encoder_layers, attr_encoder_layers = args.attr_encoder_layers,
                                  decoder_layers=args.decoder_layers, dropout=args.dropout,
                                  attention_method=args.attention_method, n_heads=args.n_heads, dropout_att=args.dropout_att, att_emb_dim=args.att_emb_dim)

        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=args.decoder_lr)

        # load pre-trained word embedding
        if args.embedding_path is not None:
            all_word_embeds = {}
            for i, line in enumerate(codecs.open(args.embedding_path, 'r', 'utf-8')):
                s = line.strip().split()
                all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

            # change emb_dim
            args.emb_dim = list(all_word_embeds.values())[-1].size
            word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_map), args.emb_dim))
            for w in word_map:
                if w in all_word_embeds:
                    word_embeds[word_map[w]] = all_word_embeds[w]
                elif w.lower() in all_word_embeds:
                    word_embeds[word_map[w]] = all_word_embeds[w.lower()]
                else:
                    # <pad> <start> <end> <unk>
                    embedding_i = torch.ones(1, args.emb_dim)
                    torch.nn.init.xavier_uniform_(embedding_i)
                    word_embeds[word_map[w]] = embedding_i

            word_embeds = torch.FloatTensor(word_embeds).to(device)
            decoder.load_pretrained_embeddings(word_embeds)
            decoder.fine_tune_embeddings(args.fine_tune_embedding)
            print('Loaded {} pre-trained word embeddings.'.format(len(word_embeds)))

    else:
        checkpoint = torch.load(args.checkpoint, map_location=str(device))
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_cider = checkpoint['metrics']["Bleu_1"]
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        decoder.fine_tune_embeddings(args.fine_tune_embedding)
        # load final_args from checkpoint
        final_args = checkpoint['final_args']
        for key in final_args.keys():
            args.__setattr__(key, final_args[key])
        if args.fine_tune_encoder is True and encoder_optimizer is None:
            print("Encoder_Optimizer is None, Creating new Encoder_Optimizer!")
            encoder.fine_tune(args.fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=args.encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    print("encoder_layers {} decoder_layers {} n_heads {} dropout {} attention_method {} encoder_lr {} "
          "decoder_lr {} alpha_c {}".format(args.encoder_layers, args.decoder_layers, args.n_heads, args.dropout,
                                            args.attention_method, args.encoder_lr, args.decoder_lr, args.alpha_c))
    #print(encoder)
    print(decoder)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    # pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
    # If your data elements are a custom type, or your collate_fn returns a batch that is a custom type.
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=args.workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, args.epochs):

        # Decay learning rate if there is no improvement for 5 consecutive epochs, and terminate training after 25
        # 8 20
        if epochs_since_improvement == args.stop_criteria:
            print("the model has not improved in the last {} epochs".format(args.stop_criteria))
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if args.fine_tune_encoder and encoder_optimizer is not None:
                print(encoder_optimizer)
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(args, train_loader=train_loader, encoder=encoder, decoder=decoder, criterion=criterion,
              encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer, epoch=epoch)

        # One epoch's validation
        metrics = validate(args, val_loader=val_loader, encoder=encoder, decoder=decoder, criterion=criterion)
        recent_cider = metrics["Bleu_1"]

        # Check if there was an improvement
        is_best = recent_cider > best_cider
        best_cider = max(recent_cider, best_cider)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(args.data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, metrics, is_best, final_args)
