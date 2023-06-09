from collections import OrderedDict
import cPickle
import os

def prototype_state():
    state = {
        'seed': 1234,
        'level': 'DEBUG',
        'oov': '<unk>',
        'end_sym_utterance': '</s>',
        'unk_sym': 0,
        'eos_sym': 1,
        'eod_sym': 2,
        'first_speaker_sym': 3,
        'second_speaker_sym': 4,
        'third_speaker_sym': 5,
        'minor_speaker_sym': 6,
        'voice_over_sym': 7,
        'off_screen_sym': 8,
        'pause_sym': 9,
        'reset_hidden_states_between_subsequences': False,
        'maxout_out': False,
        'deep_utterance_decoder_out': True,
        'deep_dialogue_encoder_input': False,
        'sent_rec_activation': 'lambda x: T.tanh(x)',
        'dialogue_rec_activation': 'lambda x: T.tanh(x)',
        'decoder_bias_type': 'all',
        'utterance_encoder_gating': 'GRU',
        'dialogue_encoder_gating': 'GRU',
        'utterance_decoder_gating': 'GRU',
        'bidirectional_utterance_encoder': False,
        'direct_connection_between_encoders_and_decoder': False,
        'deep_direct_connection': False,
        'disable_dialogue_encoder': False,
        'collaps_to_standard_rnn': False,
        'reset_utterance_decoder_at_end_of_utterance': True,
        'reset_utterance_encoder_at_end_of_utterance': False,
        'qdim_encoder': 512,
        'qdim_decoder': 512,
        'sdim': 1000,
        'rankdim': 256,
        'add_latent_gaussian_per_utterance': False,
        'condition_latent_variable_on_dialogue_encoder': False,
        'condition_posterior_latent_variable_on_dcgm_encoder': False,
        'latent_gaussian_per_utterance_dim': 10,
        'scale_latent_gaussian_variable_variances': 10,
        'min_latent_gaussian_variable_variances': 0.01,
        'max_latent_gaussian_variable_variances': 10.0,
        'condition_decoder_only_on_latent_variable': False,
        'add_latent_piecewise_per_utterance': False,
        'gate_latent_piecewise_per_utterance': True,
        'latent_piecewise_alpha_variables': 5,
        'scale_latent_piecewise_variable_alpha_use_softplus': True,
        'scale_latent_piecewise_variable_prior_alpha': 1.0,
        'scale_latent_piecewise_variable_posterior_alpha': 1.0,
        'latent_piecewise_per_utterance_dim': 10,
        'latent_piecewise_variable_alpha_parameter_tying': False,
        'latent_piecewise_variable_alpha_parameter_tying_beta': 1.0,
        'deep_utterance_decoder_input': True,
        'train_latent_variables_with_kl_divergence_annealing': False,
        'kl_divergence_annealing_rate': 1.0 / 60000.0,
        'decoder_drop_previous_input_tokens': False,
        'decoder_drop_previous_input_tokens_rate': 0.75,
        'apply_meanfield_inference': False,
        'initialize_from_pretrained_word_embeddings': False,
        'pretrained_word_embeddings_file': '',
        'fix_pretrained_word_embeddings': False,
        'fix_encoder_parameters': False,
        'do_generate_first_utterance': False,
        'skip_utterance': False,
        'skip_utterance_predict_both': False,
        'updater': 'adam',
        'use_nce': False,
        'cutoff': 0.01,
        'lr': 0.0002,
        'patience': 20,
        'cost_threshold': 1.003,
        'bs': 80,
        'sort_k_batches': 20,
        'max_grad_steps': 80,
        'save_dir': './',
        'train_freq': 10,
        'valid_freq': 5000,
        'loop_iters': 3000000,
        'time_stop': 24 * 60 * 31,
        'minerr': -1,
        'max_len': -1,
        'normop_type': 'LN',
    }

    if state['normop_type'] == 'BN':
        state['normop_gamma_init'] = 0.1
        state['normop_gamma_min'] = 0.05
        state['normop_gamma_max'] = 10.0
        state['normop_moving_average_const'] = 0.99
        state['normop_max_enc_seq'] = 50
    else:
        state['normop_gamma_init'] = 1.0
        state['normop_gamma_min'] = 0.05
        state['normop_gamma_max'] = 10.0
        state['normop_moving_average_const'] = 0.99
        state['normop_max_enc_seq'] = 1

    # Parameters for initializing the training data iterator.
    # The first is the first offset position in the list examples.
    # The second is the number of reshuffles to perform at the beginning.
    state['train_iterator_offset'] = 0
    state['train_iterator_reshuffle_count'] = 1

    return state

def prototype_test():
    state = prototype_state()
    
    # Fill paths here! 
    state['train_dialogues'] = "./tests/data/ttrain.dialogues.pkl"
    state['test_dialogues'] = "./tests/data/ttest.dialogues.pkl"
    state['valid_dialogues'] = "./tests/data/tvalid.dialogues.pkl"
    state['dictionary'] = "./tests/data/ttrain.dict.pkl"
    state['save_dir'] = "./tests/models/"

    state['max_grad_steps'] = 20
    
    # Handle pretrained word embeddings. Using this requires rankdim=10
    state['initialize_from_pretrained_word_embeddings'] = False
    state['pretrained_word_embeddings_file'] = './tests/data/MT_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = False
    
    state['valid_freq'] = 50
    
    state['prefix'] = "testmodel_" 
    state['updater'] = 'adam'
    
    state['maxout_out'] = False
    state['deep_utterance_decoder_out'] = True
    state['deep_dialogue_encoder_input'] = True

    state['utterance_encoder_gating'] = 'GRU'
    state['dialogue_encoder_gating'] = 'GRU'
    state['utterance_decoder_gating'] = 'GRU'
    state['bidirectional_utterance_encoder'] = True 
    state['direct_connection_between_encoders_and_decoder'] = True

    state['bs'] = 5
    state['sort_k_batches'] = 1
    state['use_nce'] = False
    state['decoder_bias_type'] = 'all'
    
    state['qdim_encoder'] = 15
    state['qdim_decoder'] = 5
    state['sdim'] = 10
    state['rankdim'] = 10



    return state

def prototype_test_variational():
    state = prototype_state()
    
    # Fill paths here! 
    state['train_dialogues'] = "./tests/data/ttrain.dialogues.pkl"
    state['test_dialogues'] = "./tests/data/ttest.dialogues.pkl"
    state['valid_dialogues'] = "./tests/data/tvalid.dialogues.pkl"
    state['dictionary'] = "./tests/data/ttrain.dict.pkl"
    state['save_dir'] = "./tests/models/"

    state['max_grad_steps'] = 20

    # Handle pretrained word embeddings. Using this requires rankdim=10
    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = './tests/data/MT_WordEmb.pkl' 
    
    state['valid_freq'] = 5
   
    state['prefix'] = "testmodel_"
    state['updater'] = 'adam'
    
    state['maxout_out'] = False
    state['deep_utterance_decoder_out'] = True
    state['deep_dialogue_encoder_input'] = True
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['utterance_encoder_gating'] = 'GRU'
    state['dialogue_encoder_gating'] = 'GRU'
    state['utterance_decoder_gating'] = 'LSTM'

    state['bidirectional_utterance_encoder'] = False

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 5
    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['condition_posterior_latent_variable_on_dcgm_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 10

    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75


    state['bs'] = 5
    state['sort_k_batches'] = 1
    state['use_nce'] = False
    state['decoder_bias_type'] = 'all'
    
    state['qdim_encoder'] = 15
    state['qdim_decoder'] = 5
    state['sdim'] = 10
    state['rankdim'] = 10

    state['gate_latent_piecewise_per_utterance'] = False

    # KL max-trick
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['max_kl_percentage'] = 0.01

    return state


# Twitter LSTM RNNLM model used in "A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues"
# by Serban et al. (2016), with batch norm / layer norm extension.
def prototype_twitter_lstm():
    state = prototype_state()
    
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl" 
    state['save_dir'] = "Output" 

    state['max_grad_steps'] = 80
    
    state['valid_freq'] = 5000
    
    state['prefix'] = "TwitterModel_" 
    state['updater'] = 'adam'
    
    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['reset_utterance_decoder_at_end_of_utterance'] = False
    state['collaps_to_standard_rnn'] = True
 
    state['bs'] = 80 
    state['decoder_bias_type'] = 'all'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 10
    state['qdim_decoder'] = 2000
    state['sdim'] = 10
    state['rankdim'] = 400

    return state




def prototype_twitter_HRED():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80 # If out of memory, modify this!
    state['decoder_bias_type'] = 'selective' # Choose between 'first', 'all' and 'selective'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'GRU'

    return state


# Twitter HRED model, where context biases decoder using standard MLP.
def prototype_twitter_HRED_StandardBias():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80 # If out of memory, modify this!
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    return state



def prototype_twitter_VHRED():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True



    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'selective' # Choose between 'first', 'all' and 'selective'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'GRU'


    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100


    state['scale_latent_gaussian_variable_variances'] = 0.1
    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


# Twitter VHRED model, where context biases decoder using standard MLP.
# Note, this model should be pretrained as HRED model.
def prototype_twitter_VHRED_StandardBias():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True



    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'


    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100


    state['scale_latent_gaussian_variable_variances'] = 0.1
    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


# Ubuntu LSTM RNNLM model used in "A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues"
# by Serban et al. (2016), with batch norm / layer norm extension.
def prototype_ubuntu_LSTM():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False
    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['reset_utterance_decoder_at_end_of_utterance'] = False
    state['collaps_to_standard_rnn'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 10
    state['qdim_decoder'] = 2000
    state['sdim'] = 10
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM' # Supports 'None', 'GRU' and 'LSTM'

    return state



def prototype_ubuntu_HRED():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False
    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80

    state['utterance_decoder_gating'] = 'LSTM'

    state['qdim_encoder'] = 500
    state['qdim_decoder'] = 500
    state['sdim'] = 1000
    state['rankdim'] = 300

    return state



def prototype_ubuntu_VHRED():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False
    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80

    state['utterance_decoder_gating'] = 'LSTM'

    state['qdim_encoder'] = 500
    state['qdim_decoder'] = 500
    state['sdim'] = 1000
    state['rankdim'] = 300

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1
    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


### New experiments

# Large Twitter HRED model.
def prototype_twitter_HRED_Large():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100

    state['scale_latent_gaussian_variable_variances'] = 0.1
    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state




# Large Twitter HRED model.
def prototype_twitter_VHRED_Large_SkipConnections():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterData/Training.dialogues.pkl"
    state['test_dialogues'] = "../TwitterData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterData/Validation.dialogues.pkl"
    state['dictionary'] = "../TwitterData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = True
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'
    state['direct_connection_between_encoders_and_decoder'] = False
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100

    state['scale_latent_gaussian_variable_variances'] = 0.1
    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state



# Large Twitter VHRED model.
def prototype_twitter_GaussPiecewise_VHRED_NormOp():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'GRU'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state



def prototype_twitter_Gauss_VHRED_NormOp():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'GRU'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state




def prototype_twitter_HRED_NormOp():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'GRU'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 10
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 10

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state




### CLUSTER EXPERIMENTS BEGIN: BAG-OF-WORDS DECODER

def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp1():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = True
    state['scale_latent_piecewise_variable_prior_alpha'] = 0.1
    state['scale_latent_piecewise_variable_posterior_alpha'] = 0.1

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp2():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = True
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state



def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp3():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = True
    state['scale_latent_piecewise_variable_prior_alpha'] = 10.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 10.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp4():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 0.1
    state['scale_latent_piecewise_variable_posterior_alpha'] = 0.1

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp5():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state



def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp6():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 10.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 10.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state



def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp7():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = True
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp8():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state



def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp9():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 2
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp10():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp11():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp12():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 10
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp13():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75


    state['latent_piecewise_variable_alpha_parameter_tying'] = True
    state['latent_piecewise_variable_alpha_parameter_tying_beta'] = 0.1

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp14():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75


    state['latent_piecewise_variable_alpha_parameter_tying'] = True
    state['latent_piecewise_variable_alpha_parameter_tying_beta'] = 1.0

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp15():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75


    state['latent_piecewise_variable_alpha_parameter_tying'] = True
    state['latent_piecewise_variable_alpha_parameter_tying_beta'] = 10.0

    state['patience'] = 20

    return state

def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp16():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75


    state['latent_piecewise_variable_alpha_parameter_tying'] = True
    state['latent_piecewise_variable_alpha_parameter_tying_beta'] = 100.0

    state['patience'] = 20

    return state



def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp17():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75


    state['latent_piecewise_variable_alpha_parameter_tying'] = True
    state['latent_piecewise_variable_alpha_parameter_tying_beta'] = 0.01

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp18():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 5
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75


    state['latent_piecewise_variable_alpha_parameter_tying'] = True
    state['latent_piecewise_variable_alpha_parameter_tying_beta'] = 0.001

    state['patience'] = 20

    return state



def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp19():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 10
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75


    state['latent_piecewise_variable_alpha_parameter_tying'] = True
    state['latent_piecewise_variable_alpha_parameter_tying_beta'] = 0.01

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_BOW_ClusterExp20():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'BOW'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 10
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75


    state['latent_piecewise_variable_alpha_parameter_tying'] = True
    state['latent_piecewise_variable_alpha_parameter_tying_beta'] = 0.001

    state['patience'] = 20

    return state





### CLUSTER EXPERIMENTS END: BAG-OF-WORDS DECODER



### CLUSTER EXPERIMENTS BEGIN: Twitter Baselines

###
### We do hyperparameter search for LSTM:
###
# qdim_decoder = {1000, 2000, 4000}
# rankdim = 400
# lr = 0.0002
# bs = 80
# normop_type = 'LN'

def prototype_twitter_LSTM_NormOp_ClusterExp1():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 10
    state['qdim_decoder'] = 1000
    state['sdim'] = 10
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['collaps_to_standard_rnn'] = True
    state['reset_utterance_decoder_at_end_of_utterance'] = False
    state['reset_utterance_encoder_at_end_of_utterance'] = False


    state['patience'] = 20

    return state


def prototype_twitter_LSTM_NormOp_ClusterExp2():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 10
    state['qdim_decoder'] = 2000
    state['sdim'] = 10
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['collaps_to_standard_rnn'] = True
    state['reset_utterance_decoder_at_end_of_utterance'] = False
    state['reset_utterance_encoder_at_end_of_utterance'] = False

    state['patience'] = 20

    return state


def prototype_twitter_LSTM_NormOp_ClusterExp3():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 10
    state['qdim_decoder'] = 4000
    state['sdim'] = 10
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['collaps_to_standard_rnn'] = True
    state['reset_utterance_decoder_at_end_of_utterance'] = False
    state['reset_utterance_encoder_at_end_of_utterance'] = False

    state['patience'] = 20

    return state





###
### We do hyperparameter search for HRED:
###
# sdim = {500, 1000}
# qdim_encoder = {1000}
# qdim_decoder = {1000, 2000, 4000}
# rankdim = 400
# bidirectional_utterance_encoder = True
# reset_utterance_encoder_at_end_of_utterance = False
# reset_utterance_decoder_at_end_of_utterance = True
# lr = 0.0002
# bs = 80
# normop_type = 'LN'

def prototype_twitter_HRED_NormOp_ClusterExp1():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 500
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_HRED_NormOp_ClusterExp2():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_HRED_NormOp_ClusterExp3():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state



def prototype_twitter_HRED_NormOp_ClusterExp4():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 4000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_HRED_NormOp_ClusterExp5():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 2000
    state['qdim_decoder'] = 4000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state

###
### We do hyperparameter search for Gaussian VHRED:
###
# sdim = {500, 1000}
# qdim_encoder = {1000}
# qdim_decoder = {1000, 2000, 4000}
# rankdim = 400
# latent_gaussian_per_utterance_dim = {100, 300}
# bidirectional_utterance_encoder = True
# reset_utterance_encoder_at_end_of_utterance = False
# reset_utterance_decoder_at_end_of_utterance = True
# lr = 0.0002
# bs = 80
# normop_type = 'LN'

def prototype_twitter_GaussOnly_VHRED_NormOp_ClusterExp1():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 500
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    return state


def prototype_twitter_GaussOnly_VHRED_NormOp_ClusterExp2():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    return state


def prototype_twitter_GaussOnly_VHRED_NormOp_ClusterExp3():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    return state



def prototype_twitter_GaussOnly_VHRED_NormOp_ClusterExp4():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 4000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    return state

def prototype_twitter_GaussOnly_VHRED_NormOp_ClusterExp5():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 4000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 300
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    return state

###
### We do hyperparameter search for Piecewise-Gaussian VHRED:
###
# sdim = {500, 1000}
# qdim_encoder = {1000}
# qdim_decoder = {1000, 2000, 4000}
# rankdim = 400
# latent_gaussian_per_utterance_dim = {100, 300}
# latent_piecewise_per_utterance_dim = {100, 300}
# gate_latent_piecewise_per_utterance = {False, True}
# bidirectional_utterance_encoder = True
# reset_utterance_encoder_at_end_of_utterance = False
# reset_utterance_decoder_at_end_of_utterance = True
# lr = 0.0002
# bs = 80
# normop_type = 'LN'

def prototype_twitter_GaussPiecewise_VHRED_NormOp_ClusterExp1():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 500
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_ClusterExp2():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_NormOp_ClusterExp3():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    return state



def prototype_twitter_GaussPiecewise_VHRED_NormOp_ClusterExp4():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 4000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    return state



def prototype_twitter_GaussPiecewise_VHRED_NormOp_ClusterExp5():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 4000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 300
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 300
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    return state



def prototype_twitter_GaussPiecewise_VHRED_NormOp_ClusterExp6():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    state['gate_latent_piecewise_per_utterance'] = False

    return state



def prototype_twitter_GaussPiecewise_VHRED_NormOp_ClusterExp7():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 4000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    state['gate_latent_piecewise_per_utterance'] = False

    return state



def prototype_twitter_GaussPiecewise_VHRED_NormOp_ClusterExp8():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 4000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 300
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 300
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75
    state['patience'] = 20

    state['gate_latent_piecewise_per_utterance'] = False

    return state


### CLUSTER EXPERIMENTS END: Twitter Baselines




def prototype_twitter_Gauss_VHRED_LSTMDecoder_NormOp():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state


def prototype_twitter_GaussPiecewise_VHRED_LSTMDecoder_NormOp():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state






### CLUSTER EXPERIMENTS BEGIN: Ubuntu HRED Baseline

### To be executed on the Guillumin Cluster...

###
### We do hyperparameter search for HRED without activity-entity features:
###
# sdim = {500, 1000}
# qdim_encoder = {500, 1000}
# qdim_decoder = {1000, 2000}
# rankdim = 400
# bidirectional_utterance_encoder = True
# reset_utterance_encoder_at_end_of_utterance = False
# reset_utterance_decoder_at_end_of_utterance = True
# lr = 0.0002
# bs = 80
# normop_type = {'LN', 'NONE'}



def prototype_ubuntu_HRED_NormOp_ClusterExp1():
    state = prototype_state()

    state['end_sym_utterance'] = '__eou__' # Token index 10
    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 10 # end-of-utterance symbol, which corresponds to '__eou__'. 
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    if 'UBUNTU_DATA_BPE' in os.environ:
        state['train_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Train.dialogues.pkl"
        state['test_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Test.dialogues.pkl"
        state['valid_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Valid.dialogues.pkl"
        state['dictionary'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Dataset.dict.pkl"
    else:
        state['train_dialogues'] = "../UbuntuDataBPE/Train.dialogues.pkl"
        state['test_dialogues'] = "../UbuntuDataBPE/Test.dialogues.pkl"
        state['valid_dialogues'] = "../UbuntuDataBPE/Valid.dialogues.pkl"
        state['dictionary'] = "../UbuntuDataBPE/Dataset.dict.pkl"

    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 500
    state['qdim_decoder'] = 1000
    state['sdim'] = 500
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['reset_utterance_encoder_at_end_of_utterance'] = False
    state['reset_utterance_decoder_at_end_of_utterance'] = True

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state



def prototype_ubuntu_HRED_NormOp_ClusterExp2():
    state = prototype_state()

    state['end_sym_utterance'] = '__eou__' # Token index 10
    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 10 # end-of-utterance symbol, which corresponds to '__eou__'. 
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    if 'UBUNTU_DATA_BPE' in os.environ:
        state['train_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Train.dialogues.pkl"
        state['test_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Test.dialogues.pkl"
        state['valid_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Valid.dialogues.pkl"
        state['dictionary'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Dataset.dict.pkl"
    else:
        state['train_dialogues'] = "../UbuntuDataBPE/Train.dialogues.pkl"
        state['test_dialogues'] = "../UbuntuDataBPE/Test.dialogues.pkl"
        state['valid_dialogues'] = "../UbuntuDataBPE/Valid.dialogues.pkl"
        state['dictionary'] = "../UbuntuDataBPE/Dataset.dict.pkl"


    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['reset_utterance_encoder_at_end_of_utterance'] = False
    state['reset_utterance_decoder_at_end_of_utterance'] = True

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state



def prototype_ubuntu_HRED_NormOp_ClusterExp3():
    state = prototype_state()

    state['end_sym_utterance'] = '__eou__' # Token index 10
    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 10 # end-of-utterance symbol, which corresponds to '__eou__'. 
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    if 'UBUNTU_DATA_BPE' in os.environ:
        state['train_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Train.dialogues.pkl"
        state['test_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Test.dialogues.pkl"
        state['valid_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Valid.dialogues.pkl"
        state['dictionary'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Dataset.dict.pkl"
    else:
        state['train_dialogues'] = "../UbuntuDataBPE/Train.dialogues.pkl"
        state['test_dialogues'] = "../UbuntuDataBPE/Test.dialogues.pkl"
        state['valid_dialogues'] = "../UbuntuDataBPE/Valid.dialogues.pkl"
        state['dictionary'] = "../UbuntuDataBPE/Dataset.dict.pkl"


    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['reset_utterance_encoder_at_end_of_utterance'] = False
    state['reset_utterance_decoder_at_end_of_utterance'] = True

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    return state

def prototype_ubuntu_HRED_NormOp_ClusterExp4():
    state = prototype_state()

    state['end_sym_utterance'] = '__eou__' # Token index 10
    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 10 # end-of-utterance symbol, which corresponds to '__eou__'. 
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    if 'UBUNTU_DATA_BPE' in os.environ:
        state['train_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Train.dialogues.pkl"
        state['test_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Test.dialogues.pkl"
        state['valid_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Valid.dialogues.pkl"
        state['dictionary'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Dataset.dict.pkl"
    else:
        state['train_dialogues'] = "../UbuntuDataBPE/Train.dialogues.pkl"
        state['test_dialogues'] = "../UbuntuDataBPE/Test.dialogues.pkl"
        state['valid_dialogues'] = "../UbuntuDataBPE/Valid.dialogues.pkl"
        state['dictionary'] = "../UbuntuDataBPE/Dataset.dict.pkl"

    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 500
    state['qdim_decoder'] = 1000
    state['sdim'] = 500
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['reset_utterance_encoder_at_end_of_utterance'] = False
    state['reset_utterance_decoder_at_end_of_utterance'] = True

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['normop_type'] = 'NONE'

    state['patience'] = 20

    return state



def prototype_ubuntu_HRED_NormOp_ClusterExp5():
    state = prototype_state()

    state['end_sym_utterance'] = '__eou__' # Token index 10
    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 10 # end-of-utterance symbol, which corresponds to '__eou__'. 
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    if 'UBUNTU_DATA_BPE' in os.environ:
        state['train_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Train.dialogues.pkl"
        state['test_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Test.dialogues.pkl"
        state['valid_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Valid.dialogues.pkl"
        state['dictionary'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Dataset.dict.pkl"
    else:
        state['train_dialogues'] = "../UbuntuDataBPE/Train.dialogues.pkl"
        state['test_dialogues'] = "../UbuntuDataBPE/Test.dialogues.pkl"
        state['valid_dialogues'] = "../UbuntuDataBPE/Valid.dialogues.pkl"
        state['dictionary'] = "../UbuntuDataBPE/Dataset.dict.pkl"


    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['reset_utterance_encoder_at_end_of_utterance'] = False
    state['reset_utterance_decoder_at_end_of_utterance'] = True

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['normop_type'] = 'NONE'

    state['patience'] = 20

    return state



def prototype_ubuntu_HRED_NormOp_ClusterExp6():
    state = prototype_state()

    state['end_sym_utterance'] = '__eou__' # Token index 10
    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 10 # end-of-utterance symbol, which corresponds to '__eou__'. 
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    if 'UBUNTU_DATA_BPE' in os.environ:
        state['train_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Train.dialogues.pkl"
        state['test_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Test.dialogues.pkl"
        state['valid_dialogues'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Valid.dialogues.pkl"
        state['dictionary'] = os.environ['UBUNTU_DATA_BPE']+'/'+"Dataset.dict.pkl"
    else:
        state['train_dialogues'] = "../UbuntuDataBPE/Train.dialogues.pkl"
        state['test_dialogues'] = "../UbuntuDataBPE/Test.dialogues.pkl"
        state['valid_dialogues'] = "../UbuntuDataBPE/Valid.dialogues.pkl"
        state['dictionary'] = "../UbuntuDataBPE/Dataset.dict.pkl"


    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['reset_utterance_encoder_at_end_of_utterance'] = False
    state['reset_utterance_decoder_at_end_of_utterance'] = True

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['normop_type'] = 'NONE'

    state['patience'] = 20

    return state

### CLUSTER EXPERIMENTS END: Ubuntu HRED Baseline


def prototype_twitter_HRED_NoNormalization_ClusterExp1():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../TwitterDataBPE/Train.dialogues.pkl"
    state['test_dialogues'] = "../TwitterDataBPE/Test.dialogues.pkl"
    state['valid_dialogues'] = "../TwitterDataBPE/Valid.dialogues.pkl"
    state['dictionary'] = "../TwitterDataBPE/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "TwitterModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 500
    state['rankdim'] = 400

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['normop_type'] = 'NONE'

    state['patience'] = 20

    return state





###
### We do hyperparameter search for HRED on CreateDebate:
###
### sdim = {500, 1000}
### qdim_encoder = {500, 1000}
### qdim_decoder = {500, 1000}
### rankdim = 300
### bidirectional_utterance_encoder = False
### reset_utterance_encoder_at_end_of_utterance = False
### reset_utterance_decoder_at_end_of_utterance = True
### lr = 0.0002
### bs = 80
### normop_type = 'LN'

def prototype_debate_HRED_NormOp_ClusterExp1():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 500
    state['qdim_decoder'] = 500
    state['sdim'] = 500
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state


def prototype_debate_HRED_NormOp_ClusterExp2():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = False
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = False
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state



def prototype_debate_GaussOnly_VHRED_NormOp_ClusterExp1():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 500
    state['qdim_decoder'] = 500
    state['sdim'] = 500
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 50
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 50
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/5000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state


def prototype_debate_GaussOnly_VHRED_NormOp_ClusterExp2():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/5000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state


def prototype_debate_GaussOnly_VHRED_NormOp_ClusterExp3():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state



def prototype_debate_GaussOnly_VHRED_NormOp_ClusterExp4():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 300
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 300
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state





def prototype_debate_GaussPiecewise_VHRED_NormOp_ClusterExp1():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 500
    state['qdim_decoder'] = 500
    state['sdim'] = 500
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 50
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 50
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/5000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state


def prototype_debate_GaussPiecewise_VHRED_NormOp_ClusterExp2():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/5000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state



def prototype_debate_GaussPiecewise_VHRED_NormOp_ClusterExp3():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state


def prototype_debate_GaussPiecewise_VHRED_NormOp_ClusterExp4():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 0.1
    state['scale_latent_piecewise_variable_posterior_alpha'] = 0.1

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state


def prototype_debate_GaussPiecewise_VHRED_NormOp_ClusterExp5():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 0.1
    state['scale_latent_piecewise_variable_posterior_alpha'] = 0.1

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.50

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state






def prototype_debate_GaussPiecewise_VHRED_NormOp_ClusterExp6():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/90000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state



def prototype_debate_GaussPiecewise_VHRED_NormOp_ClusterExp7():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 300
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 300
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    return state



def prototype_debate_GaussPiecewise_VHRED_NormOp_ClusterExp8():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    # KL max-trick
    state['max_kl_percentage'] = 0.01


    return state


def prototype_debate_GaussPiecewise_VHRED_NormOp_ClusterExp9():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    # KL max-trick
    state['max_kl_percentage'] = 0.025

    return state


def prototype_debate_GaussPiecewise_VHRED_NormOp_ClusterExp10():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    # KL max-trick
    state['max_kl_percentage'] = 0.05

    return state


def prototype_debate_GaussPiecewise_VHRED_NormOp_ClusterExp11():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    # KL max-trick
    state['max_kl_percentage'] = 0.005

    return state




def prototype_debate_GaussOnly_VHRED_NormOp_ClusterExp5():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    # KL max-trick
    state['max_kl_percentage'] = 0.01

    return state


def prototype_debate_GaussOnly_VHRED_NormOp_ClusterExp6():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    # KL max-trick
    state['max_kl_percentage'] = 0.025

    return state

def prototype_debate_GaussOnly_VHRED_NormOp_ClusterExp7():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    # KL max-trick
    state['max_kl_percentage'] = 0.05

    return state

def prototype_debate_GaussOnly_VHRED_NormOp_ClusterExp8():
    state = prototype_state()

    # Fill your paths here!
    state['train_dialogues'] = "../CreateDebateData/Train.Comments.pkl"
    state['test_dialogues'] = "../CreateDebateData/Test.Comments.pkl"
    state['valid_dialogues'] = "../CreateDebateData/Valid.Comments.pkl"
    state['dictionary'] = "../CreateDebateData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 2500

    state['prefix'] = "DebateModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = False

    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 40
    state['decoder_bias_type'] = 'all' # Choose between 'first', 'all' and 'selective'

    state['direct_connection_between_encoders_and_decoder'] = True
    state['deep_direct_connection'] = False

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 1000
    state['sdim'] = 1000
    state['rankdim'] = 300

    state['utterance_decoder_gating'] = 'LSTM'

    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = False
    state['kl_divergence_annealing_rate'] = 1.0/60000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['patience'] = 20

    # Settings specific for CreateDebate dataset
    state['do_generate_first_utterance'] = False

    state['initialize_from_pretrained_word_embeddings'] = True
    state['pretrained_word_embeddings_file'] = '../CreateDebateData/Word2Vec_WordEmb.pkl'
    state['fix_pretrained_word_embeddings'] = True

    # KL max-trick
    state['max_kl_percentage'] = 0.005

    return state




###
### We do hyperparameter search for (Gaussian/Piecewise) VHRED on Ubuntu:
###
### sdim = 1000
### qdim_encoder = 1000
### qdim_decoder = 2000
### rankdim = 400
### bidirectional_utterance_encoder = True
### reset_utterance_encoder_at_end_of_utterance = False
### reset_utterance_decoder_at_end_of_utterance = True
### lr = 0.0002
### bs = 80
### normop_type = 'LN'

# This is the baseline, with which we compare the 6 experiments below
# Also compare this baseline to prototype_ubuntu_HRED_NormOp_ClusterExp6 (HRED_Exp6 on Guillimin)
def prototype_ubuntu_GaussPiecewise_NormOp_VHRED_Baseline_Exp1():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80

    state['utterance_decoder_gating'] = 'LSTM'
    state['direct_connection_between_encoders_and_decoder'] = True

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = False

    state['patience'] = 20

    return state


def prototype_ubuntu_GaussPiecewise_NormOp_VHRED_Baseline_Exp2():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80

    state['utterance_decoder_gating'] = 'LSTM'
    state['direct_connection_between_encoders_and_decoder'] = True

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = True

    state['patience'] = 20

    return state


def prototype_ubuntu_GaussPiecewise_NormOp_VHRED_Exp1():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80

    state['utterance_decoder_gating'] = 'LSTM'
    state['direct_connection_between_encoders_and_decoder'] = True

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = False

    state['patience'] = 20

    return state


def prototype_ubuntu_GaussPiecewise_NormOp_VHRED_Exp2():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80

    state['utterance_decoder_gating'] = 'LSTM'
    state['direct_connection_between_encoders_and_decoder'] = True

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = False

    state['patience'] = 20

    return state


def prototype_ubuntu_GaussPiecewise_NormOp_VHRED_Exp3():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80

    state['utterance_decoder_gating'] = 'LSTM'
    state['direct_connection_between_encoders_and_decoder'] = True

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = False

    state['patience'] = 20

    return state


def prototype_ubuntu_GaussPiecewise_NormOp_VHRED_Exp4():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80

    state['utterance_decoder_gating'] = 'LSTM'
    state['direct_connection_between_encoders_and_decoder'] = True

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = True

    state['patience'] = 20

    return state


def prototype_ubuntu_GaussPiecewise_NormOp_VHRED_Exp5():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80

    state['utterance_decoder_gating'] = 'LSTM'
    state['direct_connection_between_encoders_and_decoder'] = True

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = True

    state['patience'] = 20

    return state


def prototype_ubuntu_GaussPiecewise_NormOp_VHRED_Exp6():
    state = prototype_state()

    state['end_sym_utterance'] = '__eot__'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>

    state['train_dialogues'] = "../UbuntuData/Training.dialogues.pkl"
    state['test_dialogues'] = "../UbuntuData/Test.dialogues.pkl"
    state['valid_dialogues'] = "../UbuntuData/Validation.dialogues.pkl"
    state['dictionary'] = "../UbuntuData/Dataset.dict.pkl"
    state['save_dir'] = "Output"

    state['max_grad_steps'] = 80

    state['valid_freq'] = 5000

    state['prefix'] = "UbuntuModel_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 80

    state['utterance_decoder_gating'] = 'LSTM'
    state['direct_connection_between_encoders_and_decoder'] = True

    state['qdim_encoder'] = 1000
    state['qdim_decoder'] = 2000
    state['sdim'] = 1000
    state['rankdim'] = 400

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = True

    state['patience'] = 20

    return state






###
### We do hyperparameter search for Skip-Thought Vectors and (Gaussian/Piecewise) VHRED on BookCorpus:
###
### qdim_encoder = 1200
### qdim_decoder = 2400
### rankdim = 620
### bidirectional_utterance_encoder = True
### reset_utterance_encoder_at_end_of_utterance = False
### reset_utterance_decoder_at_end_of_utterance = True
### lr = 0.0002
### bs = 80
### normop_type = 'LN'
###
### It should take about 20 days to do one pass over the dataset.
###
### Differences from original Skip-Thought Vector model:
###
### 1. We share parameters between the forward (next utterance) and backward (previous utterance) decoders.
###    The original Skip-Thought Vector model only shared the non-linear transformation applied to
###    the GRU hidden states word embedding output matrix (the GRUs had different parameters).
### 2. In the original Skip-Thought Vector model, examples for the context and both the
###    forward and backward decoder targets were given at the same time. In our model this order
###    is not preserved. Our data iterator only enforces that two adjacent sentences are
###    given to the model in the same batch most of the time.
### 3. In our model, we compute the output probablities in the decoder by applying a linear transformation
###    to both the GRU decoder hidden state and the utterance encoder hidden state. In the original
###    Skip-Thought Vector model, the linear transformation is only applied on the GRU decoder hidden state.
### 4. Optimization hyper-parameters differ. For example, original model clipped gradient norms greater
###    than 10.0, while we clip at 0.01, which is more stable with variational auto-encoders.
### 5. In original Skip-Thought Vector model sentences longer than 30 words were ignored.
###    In our model, sentence pairs longer than 60 words are ignored.
###
### None of these differences should affect performance severely. In fact, I hope they will improve performance.

# These are the three baselines, with which we compare the 6 experiments below
def prototype_book_SkipThought_NormOp():
    state = prototype_state()

    state['end_sym_utterance'] = '</s>'

    state['unk_sym'] = 0 # Unknown word token <unk>
    state['eos_sym'] = 1 # end-of-utterance symbol </s>
    state['eod_sym'] = -1 # end-of-dialogue symbol </d>
    state['first_speaker_sym'] = -1 # first speaker symbol <first_speaker>
    state['second_speaker_sym'] = -1 # second speaker symbol <second_speaker>
    state['third_speaker_sym'] = -1 # third speaker symbol <third_speaker>
    state['minor_speaker_sym'] = -1 # minor speaker symbol <minor_speaker>
    state['voice_over_sym'] = -1 # voice over symbol <voice_over>
    state['off_screen_sym'] = -1 # off screen symbol <off_screen>
    state['pause_sym'] = -1 # pause symbol <pause>


    if 'BOOK_DATA' in os.environ:
        state['train_dialogues'] = os.environ['BOOK_DATA']+'/'+"Train.pairs.pkl"
        state['test_dialogues'] = os.environ['BOOK_DATA']+'/'+"Test.mini.pairs.pkl"
        state['valid_dialogues'] = os.environ['BOOK_DATA']+'/'+"Valid.mini.pairs.pkl"
        state['dictionary'] = os.environ['BOOK_DATA']+'/'+"Dataset.dict.pkl"
    else:
        state['train_dialogues'] = "../BookCorpus/Train.pairs.pkl"
        state['test_dialogues'] = "../BookCorpus/Test.mini.pairs.pkl"
        state['valid_dialogues'] = "../BookCorpus/Valid.mini.pairs.pkl"
        state['dictionary'] = "../BookCorpus/Dataset.dict.pkl"

    state['save_dir'] = "Output"

    state['valid_freq'] = 5000

    state['prefix'] = "BookModel_SkipThought_"
    state['updater'] = 'adam'

    state['bidirectional_utterance_encoder'] = True
    state['deep_dialogue_encoder_input'] = False
    state['deep_utterance_decoder_out'] = True

    state['bs'] = 120
    state['max_grad_steps'] = 62
    state['max_len'] = 60

    state['utterance_decoder_gating'] = 'GRU'
    state['direct_connection_between_encoders_and_decoder'] = True
    state['disable_dialogue_encoder'] = True

    state['qdim_encoder'] = 1200
    state['qdim_decoder'] = 2400
    state['sdim'] = 10
    state['rankdim'] = 620

    state['do_generate_first_utterance'] = False
    state['skip_utterance'] = True
    state['skip_utterance_predict_both'] = True

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = False

    state['patience'] = 100

    return state


def prototype_book_SkipThought_NormOp_Baseline_Exp1():
    state = prototype_book_SkipThought_NormOp()

    state['qdim_encoder'] = 2400
    state['bidirectional_utterance_encoder'] = False

    return state


def prototype_book_SkipThought_NormOp_Baseline_Exp2():
    return prototype_book_SkipThought_NormOp()


def prototype_book_SkipThought_NormOp_Baseline_Exp3():
    state = prototype_book_SkipThought_NormOp()
    state['deep_utterance_decoder_input'] = True

    return state


def prototype_book_SkipThought_NormOp_VAE_Exp1():
    state = prototype_book_SkipThought_NormOp()

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = False

    return state


def prototype_book_SkipThought_NormOp_VAE_Exp2():
    state = prototype_book_SkipThought_NormOp()

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = False

    return state


def prototype_book_SkipThought_NormOp_VAE_Exp3():
    state = prototype_book_SkipThought_NormOp()

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = False

    return state


def prototype_book_SkipThought_NormOp_VAE_Exp4():
    state = prototype_book_SkipThought_NormOp()

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = False
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = True

    return state


def prototype_book_SkipThought_NormOp_VAE_Exp5():
    state = prototype_book_SkipThought_NormOp()

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = False
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = True

    return state


def prototype_book_SkipThought_NormOp_VAE_Exp6():
    state = prototype_book_SkipThought_NormOp()

    # Latent variable configuration
    state['add_latent_gaussian_per_utterance'] = True
    state['latent_gaussian_per_utterance_dim'] = 100
    state['scale_latent_gaussian_variable_variances'] = 0.1

    state['add_latent_piecewise_per_utterance'] = True
    state['latent_piecewise_per_utterance_dim'] = 100
    state['latent_piecewise_alpha_variables'] = 3
    state['scale_latent_piecewise_variable_alpha_use_softplus'] = False
    state['scale_latent_piecewise_variable_prior_alpha'] = 1.0
    state['scale_latent_piecewise_variable_posterior_alpha'] = 1.0

    state['condition_latent_variable_on_dialogue_encoder'] = True
    state['train_latent_variables_with_kl_divergence_annealing'] = True
    state['kl_divergence_annealing_rate'] = 1.0/75000.0
    state['decoder_drop_previous_input_tokens'] = True
    state['decoder_drop_previous_input_tokens_rate'] = 0.75

    state['deep_utterance_decoder_input'] = True

    return state
