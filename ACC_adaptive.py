import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn import Parameter, Softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
channel_number = 512


class ScaledDotProductAttention(nn.Module):
    def __init__(self, QKVdim):
        super(ScaledDotProductAttention, self).__init__()
        self.QKVdim = QKVdim

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: [batch_size, n_heads, -1(len_q), QKVdim]
        :param K, V: [batch_size, n_heads, -1(len_k=len_v), QKVdim]
        :param attn_mask: [batch_size, n_heads, len_q, len_k]
        """
        # scores: [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.QKVdim)
        # Fills elements of self tensor with value where mask is True.
        scores.to(device).masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_heads, len_q, len_k]
        context = torch.matmul(attn, V).to(device)  # [batch_size, n_heads, len_q, QKVdim]
        return context, attn

class Multi_Head_Attention(nn.Module):
    def __init__(self, Q_dim, K_dim, QKVdim, n_heads=8, dropout=0.1):
        super(Multi_Head_Attention, self).__init__()
        self.W_Q = nn.Linear(Q_dim, QKVdim * n_heads).to(device)
        self.W_K = nn.Linear(K_dim, QKVdim * n_heads).to(device)
        self.W_V = nn.Linear(K_dim, QKVdim * n_heads).to(device)
        self.n_heads = n_heads
        self.QKVdim = QKVdim
        self.embed_dim = Q_dim
        self.dropout = nn.Dropout(p=dropout)
        self.W_O = nn.Linear(self.n_heads * self.QKVdim, self.embed_dim).to(device)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        # q_s: [batch_size, n_heads=8, len_q, QKVdim] k_s/v_s: [batch_size, n_heads=8, len_k, QKVdim]
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        # attn_mask: [batch_size, self.n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # attn: [batch_size, n_heads, len_q, len_k]
        # context: [batch_size, n_heads, len_q, QKVdim]
        context, attn = ScaledDotProductAttention(self.QKVdim)(q_s, k_s, v_s, attn_mask)
        # context: [batch_size, n_heads, len_q, QKVdim] -> [batch_size, len_q, n_heads * QKVdim]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.QKVdim).to(device)
        # output: [batch_size, len_q, embed_dim]
        output = self.W_O(context)
        output = self.dropout(output)
        return nn.LayerNorm(self.embed_dim).to(device)(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    "Implements FFN equation."
    def __init__(self, embed_dim, d_ff, dropout):
        super(PoswiseFeedForwardNet, self).__init__()
        self.w_1 = nn.Linear(embed_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def forward(self, x):
        residual = x
        output = self.w_2(self.dropout(F.relu(self.w_1(x))))
        return nn.LayerNorm(self.embed_dim).to(device)(output + residual)
        
class LayerNorm(nn.Module):
    "Construct a layernorm module."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class LocalRNN(nn.Module):
    def __init__(self, input_dim, output_dim, ksize, dropout): #512, 512, 3
        super(LocalRNN, self).__init__()
        """
        LocalRNN structure
        """
        self.ksize = ksize
        self.rnn = nn.LSTM(output_dim, output_dim, batch_first=True)

        self.output = nn.Sequential(nn.Linear(output_dim, output_dim), nn.ReLU())

        # To speed up
        idx = [i for j in range(self.ksize-1,10000,1) for i in range(j-(self.ksize-1),j+1,1)]
        self.select_index = torch.LongTensor(idx).to(device)
        self.zeros = torch.zeros((self.ksize-1, input_dim)).to(device)

        self.conv = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1)#.to(device)
        self.fc = nn.Linear(256, 512)
        #self.fc1 = nn.Linear(1024,512)


    def forward(self, x):
        x = torch.nn.functional.glu(x, dim=-1) 
        x = self.conv((x.transpose(1, 2))).transpose(1,2)
        nbatches, l, input_dim = x.shape 
        x = self.get_K(x) 
        batch, l, ksize, d_model = x.shape
        h = self.rnn(x.view(-1, self.ksize, d_model))[0][:,-1,:]
        return h.view(batch, l, d_model)

    def get_K(self, x):
        batch_size, l, d_model = x.shape
        zeros = self.zeros.unsqueeze(0).repeat(batch_size, 1, 1)
        x = torch.cat((zeros, x), dim=1)
        key = torch.index_select(x, 1, self.select_index[:self.ksize*l])
        key = key.reshape(batch_size, l, self.ksize, -1)
        return key

class LocalRNNLayer(nn.Module):
    "Encoder is made up of attconv and feed forward (defined below)"
    def __init__(self, input_dim, output_dim, ksize, dropout):
        super(LocalRNNLayer, self).__init__()
        self.local_rnn = LocalRNN(input_dim, output_dim, ksize, dropout)
        self.connection = SublayerConnection(output_dim, dropout)

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        x = self.connection(x, self.local_rnn)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, dropout, attention_method, n_heads, dropout_att):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = Multi_Head_Attention(Q_dim=embed_dim, K_dim=embed_dim, QKVdim=64, n_heads=n_heads, dropout=dropout)
        if attention_method == "ByPixel":
            self.dec_enc_attn = Multi_Head_Attention(Q_dim=embed_dim, K_dim=4096, QKVdim=64, n_heads=n_heads, dropout=dropout)
            self.pos_ffn = PoswiseFeedForwardNet(embed_dim=embed_dim, d_ff=2048, dropout=dropout)
            
        self.local_rnn = LocalRNNLayer(embed_dim, output_dim=512, ksize=3, dropout=dropout)

        self.dec_att_attn = Multi_Head_Attention(Q_dim=embed_dim, K_dim=300, QKVdim=64, n_heads=n_heads, dropout=dropout_att)
        self.fc = nn.Linear(embed_dim*2, embed_dim)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.norm = LayerNorm(embed_dim)

    def forward(self, dec_inputs, enc_outputs, attr_out, dec_self_attn_mask, dec_enc_attn_mask, attr_dec_attn_mask):
        """
        :param dec_inputs: [batch_size, max_len=52, embed_dim=512]
        :param enc_outputs: [batch_size, num_pixels=196, 2048]
        :param dec_self_attn_mask: [batch_size, 52, 52]
        :param dec_enc_attn_mask: [batch_size, 52, 196]
        """
        #print(attr_out.size())
        dec_inputs = self.local_rnn(dec_inputs)
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)

        dec_outputs_image, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)

        dec_outputs_attr, dec_att_attn = self.dec_att_attn(dec_outputs, attr_out, attr_out, attr_dec_attn_mask)

        dec_all = torch.cat([dec_outputs_image, dec_outputs_image], dim=2)

        dec_all_a = self.fc(dec_all)

        ct = self.sig(dec_all_a)

        dec_outputs = self.norm((ct * self.relu(dec_outputs_image)) + ((1 - ct) * self.relu(dec_outputs_attr)))# + dec_outputs

        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, n_layers, vocab_size, embed_dim, dropout, attention_method, n_heads, dropout_att):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.tgt_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding.from_pretrained(self.get_position_embedding_table(embed_dim), freeze=True)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, dropout, attention_method, n_heads, dropout_att) for _ in range(n_layers)])
        self.projection = nn.Linear(embed_dim, vocab_size, bias=False).to(device)
        self.attention_method = attention_method

    def get_position_embedding_table(self, embed_dim):
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / embed_dim)
        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_idx) for hid_idx in range(embed_dim)]

        embedding_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(27)])
        embedding_table[:, 0::2] = np.sin(embedding_table[:, 0::2])  # dim 2i
        embedding_table[:, 1::2] = np.cos(embedding_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(embedding_table).to(device)

    def get_attn_pad_mask(self, seq_q, seq_k):
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # In wordmap, <pad>:0
        # pad_attn_mask: [batch_size, 1, len_k], one is masking
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

    def get_attn_subsequent_mask(self, seq):
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        subsequent_mask = np.triu(np.ones(attn_shape), k=1)
        subsequent_mask = torch.from_numpy(subsequent_mask).byte().to(device)
        return subsequent_mask

    def forward(self, encoder_out, encoded_captions, caption_lengths, attr_out):
        """
        :param encoder_out: [batch_size, num_pixels=196, 2048]
        :param encoded_captions: [batch_size, 52]
        :param caption_lengths: [batch_size, 1]
        """
        batch_size = encoder_out.size(0)
        # Sort input data by decreasing lengths.
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        attr_out = attr_out[sort_ind]
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()
        dec_outputs = self.tgt_emb(encoded_captions) + self.pos_emb(torch.LongTensor([list(range(27))]*batch_size).to(device))
        dec_outputs = self.dropout(dec_outputs)
        dec_self_attn_pad_mask = self.get_attn_pad_mask(encoded_captions, encoded_captions)
        dec_self_attn_subsequent_mask = self.get_attn_subsequent_mask(encoded_captions)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        if self.attention_method == "ByPixel":
            dec_enc_attn_mask = (torch.tensor(np.zeros((batch_size, 27, 196))).to(device) == torch.tensor(np.ones((batch_size, 27, 196))).to(device))
        elif self.attention_method == "ByChannel":
            dec_enc_attn_mask = (torch.tensor(np.zeros((batch_size, 27, channel_number))).to(device) == torch.tensor(np.ones((batch_size, 27, channel_number))).to(device))
        
        attr_dec_attn_mask = (torch.tensor(np.zeros((batch_size, 27, 7))).to(device) == torch.tensor(np.ones((batch_size, 27, 7))).to(device))

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # attn: [batch_size, n_heads, len_q, len_k]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, encoder_out, attr_out, dec_self_attn_mask, dec_enc_attn_mask, attr_dec_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        predictions = self.projection(dec_outputs)
        return predictions, encoded_captions, decode_lengths, sort_ind, dec_self_attns, dec_enc_attns


class EncoderLayer(nn.Module):
    def __init__(self, dropout, attention_method, n_heads):
        super(EncoderLayer, self).__init__()
        """
        In "Attention is all you need" paper, dk = dv = 64, h = 8, N=6
        """
        if attention_method == "ByPixel":
            self.enc_self_attn = Multi_Head_Attention(Q_dim=2048, K_dim=2048, QKVdim=64, n_heads=n_heads, dropout=dropout)
            self.pos_ffn = PoswiseFeedForwardNet(embed_dim=2048, d_ff=4096, dropout=dropout)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, n_layers, dropout, attention_method, n_heads):
        super(Encoder, self).__init__()
        if attention_method == "ByPixel":
            self.pos_emb = nn.Embedding.from_pretrained(self.get_position_embedding_table(), freeze=True)
        # self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([EncoderLayer(dropout, attention_method, n_heads) for _ in range(n_layers)])
        self.attention_method = attention_method

    def get_position_embedding_table(self):
        def cal_angle(position, hid_idx):
            x = position % 14
            y = position // 14
            x_enc = x / np.power(10000, hid_idx / 1024)
            y_enc = y / np.power(10000, hid_idx / 1024)
            return np.sin(x_enc), np.sin(y_enc)
        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_idx)[0] for hid_idx in range(1024)] + [cal_angle(position, hid_idx)[1] for hid_idx in range(1024)]

        embedding_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(196)])
        return torch.FloatTensor(embedding_table).to(device)

    def forward(self, encoder_out):
        """
        :param encoder_out: [batch_size, num_pixels=196, dmodel=2048]
        """
        batch_size = encoder_out.size(0)
        positions = encoder_out.size(1)
        enc_self_attn_mask = (torch.tensor(np.zeros((batch_size, positions, positions))).to(device)
                              == torch.tensor(np.ones((batch_size, positions, positions))).to(device))
        enc_self_attns = []
        for layer in self.layers:
            encoder_out, enc_self_attn = layer(encoder_out, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return encoder_out, enc_self_attns

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, spatial = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        #out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class Attr_EncoderLayer(nn.Module):
    def __init__(self, embed_dim, dropout, n_heads):
        super(Attr_EncoderLayer, self).__init__()
        self.attr_self_attn = Multi_Head_Attention(Q_dim=embed_dim, K_dim=embed_dim, QKVdim=64, n_heads=n_heads, dropout=dropout)
        self.pos_ffn = PoswiseFeedForwardNet(embed_dim=embed_dim, d_ff=2048, dropout=dropout)
        

    def forward(self, attr_inputs, attr_self_attn_mask):
        """
        :param attr_inputs: [batch_size, max_len=52, embed_dim=512]
        :param attr_self_attn_mask: [batch_size, 52, 52]
        """
        attr_inputs, attr_attn = self.attr_self_attn(attr_inputs, attr_inputs, attr_inputs, attr_self_attn_mask)
        attr_outputs = self.pos_ffn(attr_inputs)
        return attr_outputs, attr_attn

class Attr_Encoder(nn.Module):
    def __init__(self, n_layers, vocab_size_attr, embed_dim, dropout, n_heads):
        super(Attr_Encoder, self).__init__()
        self.vocab_size_attr = vocab_size_attr
        self.att_emb = nn.Embedding(vocab_size_attr, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([Attr_EncoderLayer(embed_dim, dropout, n_heads) for _ in range(n_layers)])

    def attr_get_attn_pad_mask(self, seq_q, seq_k):
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
        return pad_attn_mask.expand(batch_size, len_q, len_k)

    def forward(self, attr_out, attrlens):
        """
        :param encoded_captions: [batch_size, 52]
        :param caption_lengths: [batch_size, 1]
        """
        batch_size = attr_out.size(0)

        attrlens = attrlens.size(1)

        attr_outputs = self.att_emb(attr_out)
        attr_outputs = self.dropout(attr_outputs)
        attr_self_attn_pad_mask = self.attr_get_attn_pad_mask(attr_out, attr_out)
        attr_self_attn_mask = torch.gt(attr_self_attn_pad_mask, 0)
        attr_self_attns = []
        for layer in self.layers:
            attr_outputs, attr_self_attn = layer(attr_outputs, attr_self_attn_mask)
            attr_self_attns.append(attr_self_attn)
        return attr_outputs, attr_self_attns

class Transformer(nn.Module):
    """
    See paper 5.4: "Attention Is All You Need" - https://arxiv.org/abs/1706.03762
    "Apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
    In addition, apply dropout to the sums of the embeddings and the positional encodings in both the encoder
    and decoder stacks." (Now, we dont't apply dropout to the encoder embeddings)
    """
    def __init__(self, vocab_size, vocab_size_attr, embed_dim, encoder_layers, attr_encoder_layers, decoder_layers, dropout=0.1, attention_method="ByPixel", 
    n_heads=8, dropout_att=0.5, att_emb_dim=300, inter_channel=2048):
        super(Transformer, self).__init__()
        self.encoder = Encoder(encoder_layers, dropout, attention_method, n_heads)
        self.decoder = Decoder(decoder_layers, vocab_size, embed_dim, dropout, attention_method, n_heads, dropout_att)
        self.attr_encoder = Attr_Encoder(attr_encoder_layers, vocab_size_attr, att_emb_dim, dropout_att, n_heads)
        self.embedding = self.decoder.tgt_emb
        self.attention_method = attention_method
        self.cam_encoder = CAM_Module(196)
        self.norm = LayerNorm(inter_channel)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def forward(self, enc_inputs, encoded_captions, caption_lengths, attr_captions, attrlens):
        batch_size = enc_inputs.size(0)
        encoder_dim = enc_inputs.size(-1)
        if self.attention_method == "ByPixel":
            enc_inputs = enc_inputs.view(batch_size, -1, encoder_dim)
        elif self.attention_method == "ByChannel":
            enc_inputs = enc_inputs.view(batch_size, -1, encoder_dim).permute(0, 2, 1) 
        
        CAM_input = enc_inputs.permute(0, 2, 1) 
        CAM_out = self.cam_encoder(CAM_input).permute(0, 2, 1)

        attr_out, attr_self_attns = self.attr_encoder(attr_captions, attrlens)

        encoder_out, enc_self_attns = self.encoder(enc_inputs)


        encoder_out = torch.cat((encoder_out, CAM_out), dim =2)

        predictions, encoded_captions, decode_lengths, sort_ind, dec_self_attns, dec_enc_attns = self.decoder(encoder_out, encoded_captions, caption_lengths, attr_out)
        alphas = {"enc_self_attns": enc_self_attns, "dec_self_attns": dec_self_attns, "dec_enc_attns": dec_enc_attns}
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
