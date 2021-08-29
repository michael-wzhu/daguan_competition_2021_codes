import numpy as np

import torch
import tqdm
from torch.autograd import Variable


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
        torch.cumsum(mask, dim=1).type_as(mask) * mask
    ).long() + padding_idx


def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out


def to_item(x):
    """Converts x, possibly scalar and possibly tensor, to a Python scalar."""
    if isinstance(x, (float, int)):
        return x

    if float(torch.__version__[0:3]) < 0.4:
        assert (x.dim() == 1) and (len(x) == 1)
        return x[0]

    return x.item()



def replace_masked_values(tensor: torch.Tensor, mask: torch.Tensor, replace_with: float) -> torch.Tensor:
    """
    Replaces all masked values in ``tensor`` with ``replace_with``.  ``mask`` must be broadcastable
    to the same shape as ``tensor``. We require that ``tensor.dim() == mask.dim()``, as otherwise we
    won't know which dimensions of the mask to unsqueeze.

    This just does ``tensor.masked_fill()``, except the pytorch method fills in things with a mask
    value of 1, where we want the opposite.  You can do this in your own code with
    ``tensor.masked_fill((1 - mask).byte(), replace_with)``.
    """
    if tensor.dim() != mask.dim():
        raise ValueError("tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim()))
    return tensor.masked_fill((1 - mask).byte(), replace_with)


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.

    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.

    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a models that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def weighted_sum(matrix: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """
    Takes a matrix of vectors and a set of weights over the rows in the matrix (which we call an
    "attention" vector), and returns a weighted sum of the rows in the matrix.  This is the typical
    computation performed after an attention mechanism.

    Note that while we call this a "matrix" of vectors and an attention "vector", we also handle
    higher-order tensors.  We always sum over the second-to-last dimension of the "matrix", and we
    assume that all dimensions in the "matrix" prior to the last dimension are matched in the
    "vector".  Non-matched dimensions in the "vector" must be `directly after the batch dimension`.

    For example, say I have a "matrix" with dimensions ``(batch_size, num_queries, num_words,
    embedding_dim)``.  The attention "vector" then must have at least those dimensions, and could
    have more. Both:

        - ``(batch_size, num_queries, num_words)`` (distribution over words for each query)
        - ``(batch_size, num_documents, num_queries, num_words)`` (distribution over words in a
          query for each document)

    are valid input "vectors", producing tensors of shape:
    ``(batch_size, num_queries, embedding_dim)`` and
    ``(batch_size, num_documents, num_queries, embedding_dim)`` respectively.
    """
    # We'll special-case a few settings here, where there are efficient (but poorly-named)
    # operations in pytorch that already do the computation we need.
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)


###########
# for embedding
###########

def get_embedding_dict(fileName, skip_first_line=False):
    """
    Read Embedding Function

    Args:
        fileName : file which stores the embedding
    Returns:
        embeddings_index : a dictionary contains the mapping from word to vector
    """
    embeddings_index = {}
    with open(fileName, 'rb') as f:
        for i, line in tqdm.tqdm(enumerate(f)):

            if skip_first_line:
                if i == 0:
                    continue

            line_uni = line.strip()
            line_uni = line_uni.decode('utf-8')
            values = line_uni.split(' ')
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except:
                print(values, len(values))
            embeddings_index[word] = coefs
    return embeddings_index


def get_embedding_matrix(embed_dic, vocab_dic):
    """
    Construct embedding matrix

    Args:
        embed_dic : word-embedding dictionary
        vocab_dic : word-index dictionary
    Returns:
        embedding_matrix: return embedding matrix
    """
    if type(embed_dic) is not dict or type(vocab_dic) is not dict:
        raise TypeError('Inputs are not dictionary')
    if len(embed_dic) < 1 or len(vocab_dic) < 1:
        raise ValueError('Input dimension less than 1')
    vocab_sz = max(vocab_dic.values()) + 1
    EMBEDDING_DIM = len(list(embed_dic.values())[0])
    # embedding_matrix = np.zeros((len(vocab_dic), EMBEDDING_DIM), dtype=np.float32)
    embedding_matrix = np.random.rand(vocab_sz, EMBEDDING_DIM).astype(np.float32) * 0.05
    valid_mask = np.ones(vocab_sz, dtype=np.bool)
    for word, i in vocab_dic.items():
        embedding_vector = embed_dic.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be randomly initialized.
            embedding_matrix[i] = embedding_vector
        else:
            valid_mask[i] = False
    return embedding_matrix, valid_mask


def get_embedding_matrix_and_vocab(w2v_file, skip_first_line=True, include_special_tokens=True):
    """
    Construct embedding matrix

    Args:
        embed_dic : word-embedding dictionary
        skip_first_line : 是否跳过第一行
    Returns:
        embedding_matrix: return embedding matrix (numpy)
        embedding_matrix: return embedding matrix
    """
    embedding_dim = None

    # 先遍历一次，得到一个vocab list, 和向量的list
    vocab_list = []
    vector_list = []
    with open(w2v_file, "r", encoding="utf-8") as f_in:
        for i, line in tqdm.tqdm(enumerate(f_in)):
            if skip_first_line:
                if i == 0:
                    continue

            line = line.strip()
            if not line:
                continue

            line = line.split(" ")
            w_ = line[0]
            vec_ = line[1: ]
            vec_ = [float(w.strip()) for w in vec_]
            # print(w_, vec_)

            if embedding_dim == None:
                embedding_dim = len(vec_)
            else:
                # print(embedding_dim, len(vec_))
                assert embedding_dim == len(vec_)

            vocab_list.append(w_)
            vector_list.append(vec_)

    # 添加两个特殊的字符： PAD 和 UNK
    if include_special_tokens:
        vocab_list = ["pad", "unk"] + vocab_list
        # 随机初始化两个向量,
        pad_vec_ = (np.random.rand(embedding_dim).astype(np.float32) * 0.05).tolist()
        unk_vec_ = (np.random.rand(embedding_dim).astype(np.float32) * 0.05).tolist()
        vector_list = [pad_vec_, unk_vec_] + vector_list

    return vocab_list, vector_list







if __name__ == "__main__":
    w2v_file = "./resources/word2vec/dim_128_sg_0_hs_1_epochs_5/w2v.vectors"
    vocab_list, vector_list = get_embedding_matrix_and_vocab(w2v_file, skip_first_line=True)
    print(vocab_list)

    vocab_file = "./resources/word2vec/dim_128_sg_0_hs_1_epochs_5/vocab.txt"
    with open(vocab_file, "w", encoding="utf-8") as f:
        for w in vocab_list:
            f.write(w + "\n")

    w2v_matrix = np.asarray(vector_list)
    np.save(
        open("./resources/word2vec/dim_128_sg_0_hs_1_epochs_5/w2v_matrix.npy", "wb"),
        w2v_matrix,
    )




