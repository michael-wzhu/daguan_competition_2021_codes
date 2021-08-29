



## Results
测试集结果记录

|      模型描述      | dev macro-F1 | 
| ------------ |  -------------- | 
|          官方baseline：   word2vec + bilstm                |        0.39211486356053554,         |
|          随机word2vec + bilstm +  max-pool               |        0.5070020675775782  |                        |
|          随机word2vec + bilstm +  slf_attn_pool               |        0.49267002199706494  |                        |
|          预训练word2vec + bilstm +  max-pool               |       0.5013160507998644         |                    |
|          预训练word2vec + bilstm +  slf_attn_pool               |       0.4972518334336293         |                    |