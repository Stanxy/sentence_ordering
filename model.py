from pytorch_pretrained_bert.modeling import (
    BertPreTrainedModel as PreTrainedBertModel, # The name was changed in the new versions of pytorch_pretrained_bert
    gelu,
    BertEmbeddings,
    BertEncoder,
)
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import torch
from torch import nn
import torch.nn.functional as F
import math
import itertools


class MLP(nn.Module):
    def __init__(self, input_sizes, dropout_prob=0.2, bias=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(input_sizes)):
            self.layers.append(nn.Linear(input_sizes[i - 1], input_sizes[i], bias=bias))
        self.norm_layers = nn.ModuleList()
        if len(input_sizes) > 2:
            for i in range(1, len(input_sizes) - 1):
                self.norm_layers.append(nn.LayerNorm(input_sizes[i]))
        self.drop_out = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(self.drop_out(x))
            if i < len(self.layers) - 1:
                x = gelu(x)
                if len(self.norm_layers):
                    x = self.norm_layers[i](x)
        return x

class DGCN(nn.Module): # Directed Graph Convolution Network
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.05)

    def __init__(self, input_size):
        super(DGCN, self).__init__()
        self.diffusion = nn.Linear(input_size, input_size, bias=False)
        self.retained = nn.Linear(input_size, input_size, bias=False)
        self.apply(self.init_weights)

    def forward(self, A, x):
        D_out = torch.diag(A.sum(axis = 1))
        P = torch.inverse(D_out).mm(A)
        phi = D_out
        I = torch.eye(A.shape[0]).to(torch.device('cuda'))
        L_sym = I - 1/2 * (torch.pow(phi, 1/2).mm(P.mm(torch.pow(phi, -1/2))) \
         + torch.pow(phi, -1/2).mm(P.t().mm(torch.pow(phi, 1/2))))
        layer1_diffusion = L_sym.mm(gelu(self.diffusion(x))) # activation?
        x = gelu(self.retained(x) + layer1_diffusion)
        layer2_diffusion = L_sym.mm(gelu(self.diffusion(x)))
        self.predict = gelu(x + layer2_diffusion)
        return self.predict

class FormAdjcent(nn.Module):
    '''module used to form diracted adjcent matrix
    
    Inputs:
        `pooled_output` : a torch.FloatTensor of shape [sum(batch_size_i * passage_length_i), hidden_size] which is the output of 
        the first token of the last layer.
        `pairs_list` : the list of pairs denotes the position of vector in each row.
        `passage_length` : the list of passage length used to do strength.

    Outputs:
        `adjacent_matrix_list` : a list of torch.FloatTensor of shape n*n
    '''
    def __init__(self, config):
        super(FormAdjcent, self).__init__()
        self.weight_estimator = nn.Linear(config.hidden_size, 1)

    def forward(self, pooled_output, pairs_list, passage_length, pairs_num, epsilon=1e-4):
        weight = self.weight_estimator(pooled_output)
        normed_weight = torch.sigmoid(weight).squeeze()
        self.position_pointer = 0
        self.adjacent_matrix_list = []
        #print ("pairs_num : ", pairs_num)
        #print ("pairs_list : ", passage_length)
        for comb_num, length in zip(pairs_num, passage_length):
            passage_pairs_i = pairs_list[self.position_pointer : self.position_pointer + comb_num] # slice the pairs_list
            adjacent_matrix = torch.ones([length, length]).to(torch.device('cuda')) # initialize the adjacent_matrix

            for idx, pair in enumerate(passage_pairs_i):
                adjacent_matrix[pair[0], pair[1]] = normed_weight[self.position_pointer + idx]
            adjacent_matrix += epsilon # we want the graph to be strongly connected
            self.adjacent_matrix_list.append(adjacent_matrix)

            self.position_pointer += comb_num # Find the nexe front end
        
        return self.adjacent_matrix_list

class BottomLevelAttention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Params:
        config: a BertConfig class instance with the configuration to build a new model
        `attention_type` (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`


    Inputs:
        `sequence_output` (torch.FloatTensor): a torch.FloatTensor of shape 
        [sum(batch_size_i * passage_length_i), max_length, hidden_size] 
        which is the output final layer of bert.
        `pairs_list` : the list of pairs denotes the position of vector in each row.
        `passage_length` : the list of passage length used to do strength.
        `sep_positions` : the list of seperate positions of each sentence pair sequence


    Outputs:
        `integrated_pairs_of_sentence_list` : the list tensor pack for each passage. 
        [ torch.tensor ] * sample_num. each element is the tensor with a shape 
        [sent_num, edg_num, hidden_size]
    """

    def __init__(self, config, attention_type='general'):
        super(BottomLevelAttention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequence_output, pairs_list, passage_length, pairs_num, sep_positions):

        expanded_sample_num, max_len, hidden_size = sequence_output.size()

        self.position_pointer = 0
        self.integrated_pairs_of_sentence_list = [] # [ torch.tensor [sent_num, edg_num, hidden_size] ] * sample_num
        self.selected_masks = []
        self.attention_mask = []

        print ("Tn total, we have {} sep pairs.".format(len(sep_positions)))
        print ("The max length is {}".format(max_len))
        # print ("The num")

        for sep in sep_positions:
    
            sent1_musk = [0] + [1] * (sep[0] - 1) + [0] * (max_len - sep[0])
            sent1_musk = torch.tensor(sent1_musk).unsqueeze(dim=0)
            sent2_musk = [0] * (sep[0] + 1) + [1] * (sep[1] - sep[0] - 1) + [0] * (max_len - sep[1])
            sent2_musk = torch.tensor(sent2_musk).unsqueeze(dim=0)
            attention_mask = torch.cat([sent1_musk, sent2_musk], dim=0)
            self.attention_mask.append(attention_mask.unsqueeze(dim=0))

            selected_masks_i = torch.zeros(max_len)
            selected_masks_i[sep[0]] = 1
            selected_masks_i[sep[1]] = 1
            selected_masks_i = selected_masks_i.unsqueeze(dim=0)
            self.selected_masks.append(selected_masks_i)

        attention_masks = torch.cat(self.attention_mask, dim=0).to(torch.device('cuda')) # (expanded_sample_num, 2, max_len)

        self.selected_masks = torch.cat(self.selected_masks, dim=0)
        self.selected_masks = self.selected_masks.unsqueeze(dim=2)
        self.selected_masks = self.selected_masks == 1
        self.selected_masks = self.selected_masks.to(torch.device('cuda'))
        selected_querys = torch.masked_select(sequence_output, self.selected_masks).reshape(-1,2,hidden_size) # [expanded_sample_num, 2, hidden_size]

        expanded_sample_num, _, hidden_size = selected_querys.size()
        # max_len = sequence_output.size(1)

        if self.attention_type == "general":
            query = selected_querys.reshape(expanded_sample_num * 2, hidden_size)
            query = self.linear_in(query)
            query = query.reshape(expanded_sample_num, 2, hidden_size)

        # TODO: Include mask on PADDING_INDEX?

        # (expanded_sample_num, 2, hidden_size) * (expanded_sample_num, max_len, hidden_size) ->
        # (expanded_sample_num, 2, max_len)
        attention_scores = torch.bmm(query, sequence_output.transpose(1, 2).contiguous())
        attention_masks = attention_masks.to(dtype=torch.float)
        attention_masks = (1.0 - attention_masks) * -10000.0
        print ("dim of attention_scores ", attention_scores.size())
        print ("dim of attention_masks ", attention_masks.size())
        attention_scores += attention_masks

        # Compute weights across every sequence_output sequence
        attention_scores = attention_scores.view(expanded_sample_num * 2, max_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(expanded_sample_num, 2, max_len)

        # (expanded_sample_num, 2, max_len) * (expanded_sample_num, max_len, hidden_size) ->
        # (expanded_sample_num, 2, hidden_size)
        mix = torch.bmm(attention_weights, sequence_output)

        # concat -> (expanded_sample_num * 2, 2*hidden_size)
        # combined = torch.cat((mix, query), dim=2)
        # combined = combined.view(expanded_sample_num * 2, 2 * hidden_size)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (expanded_sample_num, 2, hidden_size)
        # output = self.linear_out(combined).view(expanded_sample_num, 2, hidden_size)
        # output = self.tanh(output)

        for idx_i, length in enumerate(passage_length):
            passage_pairs_i = pairs_list[self.position_pointer: self.position_pointer + pairs_num[idx_i]] # slice the pairs_list
            
            for idx_j, pair in enumerate(passage_pairs_i):
                sentence_tensor_i = [[]] * length # we have length nodes so we initiate length [] as the stack to store the tensors
                sentence_tensor_i[pair[0]].append(mix[self.position_pointer + idx_j][0].unsqueeze(0)) # push the tensor into the stack
                sentence_tensor_i[pair[1]].append(mix[self.position_pointer + idx_j][1].unsqueeze(0)) # 
            
            sentence_tensor_i = [torch.cat(tensor_stack, dim = 0).unsqueeze(0) for tensor_stack in sentence_tensor_i]
            
            self.integrated_pairs_of_sentence_list.append(torch.cat(sentence_tensor_i, dim = 0))
            self.position_pointer += pairs_num[idx_i]

        return self.integrated_pairs_of_sentence_list

class UpperLevelAttention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Params:
        config: a BertConfig class instance with the configuration to build a new model
        `attention_type` (str, optional): How to compute the attention score:

            * sum: :math:`score(H_j,q) = 1`
            * self_generate: :math:`score(H_j, q) = MLP(H_j^T)`

    Inputs:
        `tensor_packs` (torch.FloatTensor): the list tensor pack for each passage. 
        [ torch.tensor ] * sample_num. each element is the tensor with a shape 
        [sent_num, edg_num, hidden_size]


    Outputs:
        `node_list` : the list tensors of nodes for each passage. 
        [ torch.tensor ] * sample_num. each element is the tensor with a shape 
        [sent_num, hidden_size] 
    """

    def __init__(self, config, attention_type='self_generate'):
        super(UpperLevelAttention, self).__init__()

        if attention_type not in ['sum', 'self_generate']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'self_generate':
            self.linear_in = nn.Linear(config.hidden_size, 1, bias=False)

        self.linear_out = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, tensor_packs):
        node_list = []

        for sample in tensor_packs:
            passage_num, edge_num, hidden_size = sample.size()
            # edge_num = sample.size(1)

            if self.attention_type == "general":
                query = sample.view(passage_num * edge_num, hidden_size).to(torch.device('cuda'))
                query = self.linear_in(query)
                query = query.reshape(passage_num, edge_num)
            else:
                query = torch.ones(passage_num, edge_num).to(torch.device('cuda')) #  [passage_num, edge_num]

            attention_scores = query.unsqueeze(dim=1) # [passage_num, 1, hidden_size]

            # (passage_num, 1, hidden_size) * (passage_num, edge_num, hidden_size) ->
            # (passage_num, 1, edge_num)
            # attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

            # Compute weights across every context sequence
            attention_scores = attention_scores.view(passage_num * 1, edge_num)
            attention_weights = self.softmax(attention_scores)
            attention_weights = attention_weights.view(passage_num, 1, edge_num)

            # (passage_num, 1, edge_num) * (passage_num, edge_num, hidden_size) ->
            # (passage_num, 1, hidden_size)
            # (passage_num, hidden_size
            mix = torch.bmm(attention_weights, sample).squeeze()

            # concat -> (passage_num * 1, 2*hidden_size)
            # combined = torch.cat((mix, query), dim=2)
            # combined = combined.view(passage_num * 1, 2 * hidden_size)

            # # Apply linear_out on every 2nd dimension of concat
            # # output -> (passage_num, 1, hidden_size)
            # output = self.linear_out(combined).view(passage_num, 1, hidden_size)
            # output = self.tanh(output)
            node_list.append(mix)

        return node_list

class LocalDependencyEncoder(PreTrainedBertModel):

    """modified BERT model ("Bidirectional Embedding Representations from a Transformer").
    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [sum( batch_size_i * passage_length_i ), sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [sum( batch_size_i * passage_length_i ), sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [sum( batch_size_i * passage_length_i ), sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. 
        Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [sum( batch_size_i * passage_length_i ), sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block,
        `pooled_output`: a torch.FloatTensor of size [sum( batch_size_i * passage_length_i ), hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])
    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)
    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(LocalDependencyEncoder, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = MLP([config.hidden_size] * 3)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]

        first_token_tensor = sequence_output[:, 0]
        pooled_output = self.pooler(first_token_tensor)

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        #print ("The num of input_id is: ", input_ids.size(0))
        #print ("The num of pooled_output is", pooled_output.size(0))
        #exit()
        return sequence_output, pooled_output

class HierarchicalSentenceEncoder(nn.Module):
    '''module used to form sequence representations by a hierarchical approach

    Inputs:
        `encoded_layers` : a torch.FloatTensor of shape [sum(batch_size_i * passage_length_i), max_length, hidden_size] 
        which is the output of the hidden sequence in the last layer.
        `pooled_output ` : a torch.FloatTensor of shape [sum(batch_size_i * passage_length_i), hidden_size] which is the output of 
        the first token of the last layer.
        `pairs_list` : The list of pairs denotes the position of vector in each row.
        `passge_length` : The list of passage length used to do strength.
        `sep_positions` : The seperate positions of each sentences.
    
    Outputs:
        `nodes` : list of node tensors. tensor shape [passage_num, hidden_size]
        `graphs` : list of adjacent graph tensors. tensor shape [passage_num, passage_num]
    '''

    def __init__(self, bert_model, cache_dir=None, state_dict=None):
        super(HierarchicalSentenceEncoder, self).__init__()
        if cache_dir!=None:
            self.bert_enc = LocalDependencyEncoder.from_pretrained(bert_model, cache_dir = cache_dir)
        else:
            self.bert_enc = LocalDependencyEncoder.from_pretrained(bert_model, state_dict=state_dict)
        self.config = self.bert_enc.config
        self.graph_generator = FormAdjcent(self.config)
        self.bottom_level_encoder = BottomLevelAttention(self.config)
        self.upper_level_encoder = UpperLevelAttention(self.config)
        

    def forward(self, input_ids, token_type_ids, masked_ids, pairs_list, passage_length, pairs_num, sep_positions):
        encoded_layers, pooled_output = self.bert_enc(input_ids, token_type_ids, masked_ids)
        graphs = self.graph_generator(pooled_output, pairs_list, passage_length, pairs_num)
        encoded_sentences = self.bottom_level_encoder(encoded_layers, pairs_list, passage_length, pairs_num, sep_positions)
        nodes = self.upper_level_encoder(encoded_sentences)
        return nodes, graphs

class GlobalGraphPropagation(nn.Module):
    ''' Do graph propagation on graphs in the batch

    Inputs: 
        `graphs` : list of adjacent graph tensors. tensor shape [passage_num, passage_num]
        `nodes` : list of node tensors. tensor shape [passage_num, hidden_size]
        `passage_length` : the list of passage length used to do strength.
    Outputs: 
        `passage_represent` : tensor of the final representtion of the passage batch, padded
    '''

    def __init__(self, hidden_size):
        super(GlobalGraphPropagation, self).__init__()
        self.GNN = DGCN(hidden_size)
        self.hidden_size = hidden_size
    def forward(self, nodes, graphs, passage_length):
        node_sets = [self.GNN(graph, node) for graph, node in zip(graphs, nodes)]
        # we need to pad the sets and make it able to go batch processing
        max_passage_num = max(passage_length)
        new_batch = []
        for node_set in node_sets:
            sent_num, _ = node_set.size()
            pad_tensor = torch.zeros(max_passage_num - sent_num, self.hidden_size).to(torch.device('cuda'))
            node_set = torch.cat([node_set, pad_tensor], dim=0).unsqueeze(0)
            new_batch.append(node_set)
        return torch.cat(new_batch, dim=0)

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0: # we only want config.num_attention_heads head attention
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        
        self._inf = nn.Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) 
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) 
        # tensor.Size([batch, max_len, num_attention_heads, attention_head_size])
        x = x.view(*new_x_shape) 
        return x.permute(0, 2, 1, 3) # tensor.Size([batch, num_attention_heads, max_len, attention_head_size])

    def forward(self, hidden_states, attention_mask, cell_state, t):
        mixed_query_layer = self.query(cell_state.unsqueeze(1)) # Find the newest node 
        mixed_key_layer = self.key(hidden_states[:,:-1])
        mixed_value_layer = self.value(hidden_states[:,:-1])

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # [batch_size, head, hidden_size] * [batch_size, hidden_size, max_length]
        # [batch_size, head, max_length]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) 
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # [batch_size, num_head, max_len]
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # attention_scores = attention_scores.squeeze()
        # [batch_size, max_length]
        mask = (self.inf[:,:t] * torch.eq(attention_mask, 0).float()).unsqueeze(dim=1).unsqueeze(dim=2).expand(-1, \
            self.num_attention_heads, 1, -1)
        print ("num of head", self.num_attention_heads)
        print ("Size of mask", mask.size())
        print ("Size of attention", attention_scores.size())
        attention_scores = attention_scores + mask # [batch_size, num_head, max_length]

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # [batch_size, 1, max_length] * [batch_size, max_length, hidden_size]
        # [batch_size, 1, hidden_size]
        context_layer = torch.matmul(attention_probs, value_layer) 
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer.squeeze()

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)

class Attention(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, 
                input_dim,
                hidden_dim):
        """
        Initiate Attention
        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = nn.Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = nn.Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        # Initialize vector V
        nn.init.uniform_(self.V, -1, 1)

    def forward(self, input_,
                context,
                mask):
        """
        Attention - Forward-pass
        :param Tensor input: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """

        # (batch, hidden_dim, seq_len)
        inp = self.input_linear(input_).unsqueeze(2).expand(-1, -1, context.size(1))

        # (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)

        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # (batch, seq_len)
        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
        if len(att[mask]) > 0: # mask the unneeded ones
            att[mask] = self.inf[mask]
        alpha = self.softmax(att)

        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)

class Decoder(nn.Module):
    '''Attentive decoder architecture'''

    def __init__(self, config):
        """
        Initiate Decoder
        :param int config: a BertConfig class instance with the configuration to build a new model
        """

        super(Decoder, self).__init__()
        
        self.embedding_dim = config.hidden_size
        self.hidden_dim = config.hidden_size

        self.input_to_hidden = nn.Linear(self.embedding_dim, 4 * self.hidden_dim)
        self.hidden_to_hidden = nn.Linear(self.hidden_dim, 4 * self.hidden_dim)
        self.hidden_out_forward = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.att = Attention(self.hidden_dim, self.hidden_dim)
        self.first_input = nn.Parameter(torch.FloatTensor(self.hidden_dim), requires_grad=True) # the trainable first input
        self.backward_attention = BertSelfAttention(config)
        
        nn.init.uniform_(self.first_input, -1, 1)
        # Used for propagating .cuda() command
        self.mask = nn.Parameter(torch.ones(1).byte(), requires_grad=False)
        self.runner = nn.Parameter(torch.zeros(1).byte(), requires_grad=False)

    def step(self, x, embedded_inputs, hidden, context, mask, t):
        """
        Recurrence step function
        :param Tensor x: Input at time t
        :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
        :param Tensor context: Generated states at time t-1
        :param Tensor mask: Mask tensor at time t-1
        :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
        """

        # Regular LSTM
        h, c = hidden

        mask_step = mask[:,:t]

        if t > 0:
            b = self.backward_attention(context, mask_step, c, t) # [batch_size, hidden_size]

        print ("The size of hidden is : ", h.size())
        print ("The size of input is : ", x.size())
        gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
        input_, forget, cell, out = gates.chunk(4, 1)

        input_ = F.sigmoid(input_)
        forget = F.sigmoid(forget)
        cell = F.tanh(cell)
        out = F.sigmoid(out)

        c_t = (forget * c) + (input_ * cell)

        if t > 0:
            backward_context_t = out * x + (1 - out) * b
        else:
            backward_context_t = x
        #backward_hidden_t = self.hidden_out_backward(backward_context_t)

        # Attention section
        foward_context_t, output = self.att(backward_context_t, embedded_inputs, torch.eq(mask, 0))
            # for the sent we want we set the mask to 1, for the sent we don't want, we set them to zero
        hidden_t = F.tanh(self.hidden_out_forward(torch.cat((foward_context_t, backward_context_t), 1)))

        return hidden_t, c_t, output
    
    
    def forward(self, embedded_inputs, hidden, mask_length, answers):
        """
        Decoder - Forward-pass
        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor hidden: First decoder's hidden states
        :param Tensor mask_length: The mask tensor 
        :param Tensor answers: answers's outputs. [batch_size, max_len, max_len]
        :return: (Output probabilities, Pointers indices), last hidden state
        """

        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        # (batch, seq_len)
        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1) # all ones
        self.att.init_inf(mask.size()) # all neg inf
        mask = mask_length * mask # when the length is over the total num of sentences, we set the mask to zero

        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).byte() # [barch_size, max_len]

        decoder_input = self.first_input.unsqueeze(dim=0).expand(embedded_inputs.size(0), -1) # [batch_size, hidden_size]
        backward_mask = torch.cat([torch.ones(mask.size(0), 1).byte().to(torch.device('cuda')) \
            , mask[:,:-1]], dim=1) # move one unit left. 
        self.backward_attention.init_inf(backward_mask.size())
        past = decoder_input.unsqueeze(dim=1)

        outputs = []
        pointers = []
        
        # Recurrence loop
        for t in range(input_length):
            h_t, c_t, outs = self.step(decoder_input, embedded_inputs, hidden, past, backward_mask, t)
            hidden = (h_t, c_t)

            # Masking selected inputs
            masked_outs = outs * mask.float()

            # Get maximum probabilities and indices
            if answers is None:
                max_probs, indices = masked_outs.max(1)
            else:
                indices = answers[:,t] # batch_size, max_len, max_len 
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1]).byte()) #[batch_size, max_len]
            #print("the indices are : ", indices)
            # Update mask to ignore seen indices
            mask  = (mask.float() * (1 - one_hot_pointers.float())).byte()
            #print("mask values are : ", mask)
            # Get embedded inputs by max indices
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim).byte()#[batch_size, max_len, hidden_size]
            print("Our masks are : ", embedding_mask)
            print("Our embedded inputs are : ", embedded_inputs)
            decoder_input = embedded_inputs[embedding_mask.data].view(batch_size, self.embedding_dim) # []
            #if t == input_length -1:
            #    exit()
            past = torch.cat([past, decoder_input.unsqueeze(dim=1)], dim=1)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 2, 0) # batch_size, class_num, seq_len
        pointers = torch.cat(pointers, 1) # 

        return (outputs, pointers), hidden

class BackwardFowardAttentiveDecoder(nn.Module):
    """
    Pointer-Net
    """

    def __init__(self, config):
        """
        Initiate Pointer-Net
        :param int config: a BertConfig class instance with the configuration to build a new model
        """

        super(BackwardFowardAttentiveDecoder, self).__init__()
        self.decoder = Decoder(config)

        # Initialize decoder_input0
        # nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, inputs, passage_length, answers=None):
        """
        PointerNet - Forward-pass
        :param Tensor inputs: Input sequence
        :return: Pointers probabilities and indices
        """

        batch_size = inputs.size(0)
        input_length = inputs.size(1)

        decoder_input0 = (inputs.sum(dim=1), inputs.sum(dim=1)) # this is the hidden vector we have
        if answers != None:
            answers = self.pad_answers(passage_length, answers)
        mask_length = self.generate_mask(passage_length)
        print (inputs)
        (outputs, pointers), decoder_hidden = self.decoder(inputs,
                                                           decoder_input0,
                                                           mask_length, 
                                                           answers)
        # pointers torch.tensor [batch_size, sample_len]
        return  outputs, pointers

    def pad_answers(self, passage_length, answers):
        max_len = max(passage_length)
        answers_padded = []
        for answer in answers:
            answer += [max_len - 1] * (max_len - len(answer))
            answers_padded.append(torch.LongTensor(answer).unsqueeze(dim=0))
        return torch.cat(answers_padded, dim=0).to(torch.device('cuda')) # [batch_size, max_len]

    def generate_mask(self, passage_length):
        max_len = max(passage_length)
        mask = []
        for length in passage_length:
            mask_i = [1] * length + [0] * (max_len - length)
            mask_tensor_i = torch.ByteTensor(mask_i).unsqueeze(0)
            mask.append(mask_tensor_i)
        return torch.cat(mask, dim=0).to(torch.device('cuda')) # [batch_size, max_len]

def calculate_loss(batch, model1, model2, model3, device, critic):
    '''Function to gain Loss

    Inputs : 
        batch : training batch, contains the needed data
        model 1 : the bottom layer model
        model 2 : the middle layer model
        model 3 : the last layer model
        device : the device that takes in the datacude or cpu.
        critic : a torch.nn loss obj which used to calculate the loss
        

    Outputs :
        loss : the final loss of the training
    '''
    
    input_ids, token_type_ids, masked_ids, pairs_list, sep_positions, ground_truth, passage_length, pairs_num = batch
    #try:
    #    assert masked_ids == torch.int64
    #except:
    #    print (masked_ids)
    #    exit()
    nodes, graphs = model1(input_ids, token_type_ids, masked_ids, pairs_list, passage_length, pairs_num, sep_positions)
    encoded_nodes = model2(nodes, graphs, passage_length)
    logits_output, pointers_tensor = model3(encoded_nodes, passage_length, ground_truth)
    
    logits = torch.log(logits_output)
    C = model3.pad_answers(passage_length, ground_truth).to(dtype=torch.long).to(device)
    mask = model3.generate_mask(passage_length).byte().to(device)
    log_loss = critic(logits, C)
    log_loss_masked = log_loss.masked_fill(mask = 1-mask, value = torch.tensor(0))
    
    pointers_masked = pointers_tensor.masked_fill(mask = 1-mask, value = torch.tensor(-1)).tolist()
    pointers = [ one_pointer[:passage_length[idx]] for idx, one_pointer in enumerate(pointers_masked)]
    # now the pointers and gournd truth has the same data structure
    return log_loss_masked.sum() / logits.size(0), pointers, ground_truth

def dev_test(batch, model1, model2, model3, device):
    '''Function to gain Loss

    Inputs : 
        batch : testing batch, contains the needed data. One should be aware that the testing batch size is constantly 1
        model 1 : the bottom layer model
        model 2 : the middle layer model
        model 3 : the last layer model
        device : the device that takes in the datacude or cpu.

    Outputs :
        best_pointers : the selected ordering list
        ground_truth : the true ordering list
    '''

    input_ids, token_type_ids, masked_ids, pairs_list, sep_positions, ground_truth, passage_length, pairs_num = batch
    nodes, graphs = model1(input_ids, token_type_ids, masked_ids, pairs_list, passage_length, pairs_num, sep_positions)
    encoded_nodes = model2(nodes, graphs, passage_length)
    # _, pointers_tensor = model3(inputs, passage_length)
    best_pointers = beam_search_pointer(model3, encoded_nodes, passage_length)
    # logits = torch.log(logits_output)
    # mask = model3.generate_mask(passage_length).byte().to(device)

    #pointers_masked = pointers_tensor.masked_fill(mask = 1-mask, value = torch.tensor(-1)).tolist()
    # pointers = [ one_pointer[:one_pointer.index(-1)] for one_pointer in pointers_mask]

    return best_pointers, ground_truth
    
def predict(batch, model1, model2, model3, device, critic):
    '''Function to gain Loss

    Inputs : 
        batch : training batch, contains the needed data
        model 1 : the bottom layer model
        model 2 : the middle layer model
        model 3 : the last layer model
        device : the device that takes in the datacude or cpu.
        critic : a torch.nn loss obj which used to calculate the loss
        

    Outputs :
        loss : the final loss of the training
    '''

    input_ids, token_type_ids, masked_ids, pairs_list, sep_positions, passage_length, pairs_num = batch
    nodes, graphs = model1(input_ids, token_type_ids, masked_ids, pairs_list, passage_length, pairs_num, sep_positions)
    encoded_nodes = model2(nodes, graphs, passage_length)
    best_pointers = beam_search_pointer(model3, encoded_nodes, passage_length)
    # logits = torch.log(logits_output)
    # mask = model3.generate_mask(passage_length).byte().to(device)

    #pointers_masked = pointers_tensor.masked_fill(mask = 1-mask, value = torch.tensor(-1)).tolist()
    # pointers = [ one_pointer[:one_pointer.index(-1)] for one_pointer in pointers_mask]

    return best_pointers

class Beam(object):
    def __init__(self, beam_size):
        self.beam_size = beam_size

        self.candidates = []
        self.scores = []

    def step(self, prob, prev_beam, f_done):
        pre_score = prob.new_tensor(prev_beam.scores) # tensor [len] ??

        score = prob + pre_score.unsqueeze(-1).expand_as(prob) # [1, max_len]
        if score.numel() < self.beam_size:
            nbest_score, nbest_ix = score.view(-1).topk(score.numel(), largest=False) # The kth smallest element are selected
        else:
            nbest_score, nbest_ix = score.view(-1).topk(self.beam_size, largest=False) # The kth smallest element are selecte

        # with topk we got a ranked score
        beam_ix = nbest_ix / prob.size(1) # find the divisible factor. long type tensor will fint the mod.
        token_ix = nbest_ix - beam_ix * prob.size(1) # find the token

        done_list, remain_list = [], []
        prev_candidates = prev_beam.candidates
        for b_score, b_ix, t_ix in itertools.zip_longest(nbest_score.tolist(), beam_ix.tolist(), token_ix.tolist()): # at least num k
            candidate = prev_candidates[b_ix] + [t_ix] # put the token into the list - candidate = [token1]
            # list of the tokens. in this way the prev_candidates are able to pro_length 

            if f_done(candidate): # if the length of candidate is equal to the length of the passage
                done_list.append([candidate, b_score]) # append [candidate and the score into the done list. Means we finish this sample
            else: # if we still do not finish it yet
                remain_list.append(b_ix) # beam b_ix has not been finished
                self.candidates.append(candidate) # update candates [ [token1] ]
                self.scores.append(b_score) # update scores [ most likely token ]
        return done_list, remain_list # done list is a list of list. remain_list stores all unfinished index

def beam_search_pointer(model, encoded_nodes, passage_length, beam_size = 32):
    '''
    beam search pointer used in prediction.

    Inputs:
        model : the model contains the decoder step cell
        encoded_nodes : the encoded nodes as input
        passage_length : the list of the length of the passages
        beam_size : the size of beam search
    Outputs:

    '''
    # sentences, _, dec_init, keys = model.encode(encoded_nodes, doc_num, ewords_and_len, elocs)

    document = encoded_nodes.squeeze(0) # [T, H]
    T, H = document.size()

    W = beam_size

    prev_beam = Beam(W)
    prev_beam.candidates = [[]] # list of list
    prev_beam.scores = [0]

    # target_t = T - 1
    target_t = T

    f_done = (lambda x: len(x) == target_t) # the flag to distinguish done and undone beam

    valid_size = W # the beam weight
    hyp_list = [] # 

    # mask = model.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1) # all ones
    # model.att.init_inf(mask.size()) # all neg inf
    # mask = mask_length * mask # when the length is over the total num of sentences, we set the mask to zero
    remain_list = [0]
    for t in range(target_t):
        candidates = prev_beam.candidates # candidates  = [ [idx11, idx12, ...], [idx21, idx22, ...] ] 
        if t == 0:
            # start
            # dec_input = encoded_nodes.new_zeros(1, 1, H) # tensor size [batch_size, 1, hidden_size]. Here the 1 is perhaps used foe broadcasting
            mask = encoded_nodes.new_zeros(1, T).byte() # (byte) tensor size [batch_size, max_lenth]. Here 1 is perhaps used for broadcasting
            decoder_input = model.first_input.unsqueeze(dim=0).expand(encoded_nodes.size(0), -1)# .unsqueeze(dim=1) # [batch_size, hidden_size]
            backward_mask = torch.cat([torch.ones(mask.size(0), 1), mask[:,-1]], dim=1).byte() # move one unit left. 
            model.backward_attention.init_inf(backward_mask.size())
            past = decoder_input.unsqueeze(dim=1)
            hidden = (encoded_nodes.sum(dim=1), encoded_nodes.sum(dim=1))
        else:
            # select the last list element in each element list of candidates.  
            index = encoded_nodes.new_tensor(list(map(lambda cand: cand[-1], candidates))).long() # [beam] long teensor
            # beam 1 H
            decoder_input = document[index]# .unsqueeze(1) # [beam, hidden size] , the second dimension is added after subtract the beam.

            mask[torch.arange(index.size(0)), index] = 1 # [beam_size, max_len]

            beam_remain_ix = encoded_nodes[0].new_tensor(remain_list)

            past = past.index_select(0, beam_remain_ix)

            past = torch.cat([past, decoder_input.unsqueeze(dim=1)], dim=1) # [beam_size, max_len]

        h_t, c_t, outs = model.step(decoder_input, encoded_nodes, hidden, past, backward_mask, t) # outs - float tensor [beam_size, max_len]
        logp = torch.log(outs)
        next_beam = Beam(valid_size) # generat another beam 
        done_list, remain_list = next_beam.step(-logp, prev_beam, f_done) # 
        if t == 0:
            beam_remain_ix = encoded_nodes[0].new_tensor(remain_list)
        hyp_list.extend(done_list)
        valid_size -= len(done_list) # we have finished some of the sequence

        if valid_size == 0: # when we find we have generated beam sequence, we shall break the loop
            break

         # remain list [num] 
        h_t = h_t.index_select(1, beam_remain_ix) # 
        c_t = c_t.index_select(1, beam_remain_ix) # 
        hidden = (h_t, c_t)

        mask = mask.index_select(0, beam_remain_ix)

        backward_mask = backward_mask.index_select(0, beam_remain_ix)

        prev_beam = next_beam

    score = h_t.new_tensor([hyp[1] for hyp in hyp_list]) # the 1 position of hyp is the score, zero is the ... ?
    sort_score, sort_ix = torch.sort(score) # sort_ix is the pointer of each rank in the original score.
    output = []
    for ix in sort_ix.tolist():
        output.append((hyp_list[ix][0], score[ix].item()))
    best_output = output[0][0] # the zero and zero 

    # the_last = list(set(list(range(T))).difference(set(best_output)))
    # best_output.append(the_last[0])

    return best_output