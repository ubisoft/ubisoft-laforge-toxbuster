from transformers.models.bert.configuration_bert import BertConfig


class BertWithSpeakerSegmentationConfig(BertConfig):
    '''
    Bert Config + details for speaker segmentation
    '''

    def __init__(self,
                 vocab_size=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 pad_token_id=0,
                 position_embedding_type="absolute",
                 use_cache=True,
                 classifier_dropout=None,
                 player_ID_vocab_size = 13, # 0 for special token, P1 - P12
                 chat_type_vocab_size = 3,  # 0 for special token, 1 -> ALL_CHAT, 2 -> TEAM_CHAT
                 team_ID_vocab_size = 3,    # 0 for special token, T1 - T2
                 **kwargs):

        # Initialize Bert Config
        super().__init__(vocab_size,
                         hidden_size,
                         num_hidden_layers,
                         num_attention_heads,
                         intermediate_size,
                         hidden_act,
                         hidden_dropout_prob,
                         attention_probs_dropout_prob,
                         max_position_embeddings,
                         type_vocab_size,
                         initializer_range,
                         layer_norm_eps,
                         pad_token_id,
                         position_embedding_type,
                         use_cache,
                         classifier_dropout,
                         **kwargs)

        # Add our own config
        self.player_ID_vocab_size = player_ID_vocab_size
        self.chat_type_vocab_size = chat_type_vocab_size
        self.team_ID_vocab_size = team_ID_vocab_size

