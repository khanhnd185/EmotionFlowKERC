from config import *
from model import CRFModel

speaker_vocab_dict_path = 'vocabs/speaker_vocab.pkl'
emotion_vocab_dict_path = 'vocabs/emotion_vocab.pkl'


def pad_to_len(list_data, max_len, pad_value):
    list_data = list_data[-max_len:]
    len_to_pad = max_len-len(list_data)
    pads = [pad_value] * len_to_pad
    list_data.extend(pads)
    return list_data

def load_kerc_and_builddataset(file_path, train=False):
    speaker_vocab = vocab.UnkVocab.from_dict(torch.load(
        speaker_vocab_dict_path
    ))
    emotion_vocab = vocab.Vocab.from_dict(torch.load(
        emotion_vocab_dict_path
    ))

    data = pd.read_csv(file_path)
    ret_utterances = []
    ret_speaker_ids = []
    ret_emotion_idxs = []
    utterances = []
    full_contexts = []
    speaker_ids = []
    emotion_idxs = []
    pre_dial_id = -1
    max_turns = 0
    for row in tqdm(data.iterrows(), desc='processing file {}'.format(file_path)):
        meta = row[1]
        utterance = meta['Utterance'].replace(
            '’', '\'').replace("\"", '')
        speaker = meta['Speaker']
        utterance = speaker + ' 이, ' + utterance + ', 라고 말했다'
        emotion = meta['Emotion'].lower()
        dialogue_id = meta['Dialogue_ID']
        utterance_id = meta['Utterance_ID']
        if pre_dial_id == -1:
            pre_dial_id = dialogue_id
        if dialogue_id != pre_dial_id:
            ret_utterances.append(full_contexts)
            ret_speaker_ids.append(speaker_ids)
            ret_emotion_idxs.append(emotion_idxs)
            max_turns = max(max_turns, len(utterances))
            utterances = []
            full_contexts = []
            speaker_ids = []
            emotion_idxs = []
        pre_dial_id = dialogue_id
        speaker_id = speaker_vocab.word2index(speaker)
        emotion_idx = utterance_id
        token_ids = tokenizer(utterance, add_special_tokens=False)[
            'input_ids'] + [CONFIG['SEP']]
        full_context = []
        if len(utterances) > 0:
            context = utterances[-3:]
            for pre_uttr in context:
                full_context += pre_uttr
        full_context += token_ids
        # query
        query = '지금 ' + speaker + ' 은 [MASK]'
        query_ids = tokenizer(query, add_special_tokens=False)['input_ids'] + [CONFIG['SEP']]
        full_context += query_ids

        full_context = pad_to_len(
            full_context, CONFIG['max_len'], CONFIG['pad_value'])
        # + CONFIG['shift']
        utterances.append(token_ids)
        full_contexts.append(full_context)
        speaker_ids.append(speaker_id)
        emotion_idxs.append(emotion_idx)

    pad_utterance = [CONFIG['SEP']] + tokenizer(
        "1",
        add_special_tokens=False
    )['input_ids'] + [CONFIG['SEP']]
    pad_utterance = pad_to_len(
        pad_utterance, CONFIG['max_len'], CONFIG['pad_value'])
    # for CRF
    ret_mask = []
    ret_last_turns = []
    for dial_id, utterances in tqdm(enumerate(ret_utterances), desc='build dataset'):
        mask = [1] * len(utterances)
        while len(utterances) < max_turns:
            utterances.append(pad_utterance)
            ret_emotion_idxs[dial_id].append(-1)
            ret_speaker_ids[dial_id].append(0)
            mask.append(0)
        ret_mask.append(mask)
        ret_utterances[dial_id] = utterances

        last_turns = [-1] * max_turns
        for turn_id in range(max_turns):
            curr_spk = ret_speaker_ids[dial_id][turn_id]
            if curr_spk == 0:
                break
            for idx in range(0, turn_id):
                if curr_spk == ret_speaker_ids[dial_id][idx]:
                    last_turns[turn_id] = idx
        ret_last_turns.append(last_turns)
    dataset = TensorDataset(
        torch.LongTensor(ret_utterances),
        torch.LongTensor(ret_speaker_ids),
        torch.LongTensor(ret_emotion_idxs),
        torch.ByteTensor(ret_mask),
        torch.LongTensor(ret_last_turns)
    )
    return dataset

def test(model, data_path):
    data = load_kerc_and_builddataset(data_path)

    pred_list = []
    hidden_pred_list = []
    selection_list = []
    utterance_idxs_list = []
    model.eval()
    sampler = SequentialSampler(data)
    dataloader = DataLoader(
        data,
        batch_size=CONFIG['batch_size'],
        sampler=sampler,
        num_workers=0,  # multiprocessing.cpu_count()
    )
    tq_test = tqdm(total=len(dataloader), desc="testing", position=2)
    with torch.no_grad():
        for batch_id, batch_data in enumerate(dataloader):
            batch_data = [x.to(model.device()) for x in batch_data]
            sentences = batch_data[0]
            speaker_ids = batch_data[1]
            utterance_idxs = batch_data[2].cpu().numpy().tolist()
            mask = batch_data[3]
            last_turns = batch_data[4]
            outputs = model(sentences, mask, speaker_ids, last_turns)
            for batch_idx in range(mask.shape[0]):
                for seq_idx in range(mask.shape[1]):
                    if mask[batch_idx][seq_idx]:
                        pred_list.append(outputs[batch_idx][seq_idx])
                        utterance_idxs_list.append(utterance_idxs[batch_idx][seq_idx])
            tq_test.update()
    submit(utterance_idxs_list, pred_list, filename='submit.csv')

def submit(id_list, pred_list, filename):
    emotion_vocab = vocab.Vocab.from_dict(torch.load(
        emotion_vocab_dict_path
    ))

    with open(filename, 'w') as f:
        f.write("Id,Predicted\n")

        for id, pred in zip(id_list, pred_list):
            f.write("{},{}\n".format(id, emotion_vocab.index2word(pred)))

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-te', '--test', action='store_true',
                        help='run test', default=False)
    parser.add_argument('-tr', '--train', action='store_true',
                        help='run train', default=True)
    parser.add_argument('-ft', '--finetune', action='store_true',
                        help='fine tune base the best model', default=False)
    parser.add_argument('-pr', '--print_error', action='store_true',
                        help='print error case', default=False)
    parser.add_argument('-bsz', '--batch', help='Batch_size',
                        required=False, default=CONFIG['batch_size'], type=int)
    parser.add_argument('-epochs', '--epochs', help='epochs',
                        required=False, default=CONFIG['epochs'], type=int)
    parser.add_argument('-lr', '--lr', help='learning rate',
                        required=False, default=CONFIG['lr'], type=float)
    parser.add_argument('-p_unk', '--p_unk', help='prob to generate unk speaker',
                        required=False, default=CONFIG['p_unk'], type=float)
    parser.add_argument('-ptmlr', '--ptm_lr', help='ptm learning rate',
                        required=False, default=CONFIG['ptmlr'], type=float)
    parser.add_argument('-tsk', '--task_name', default='kerc', type=str)
    parser.add_argument('-fp16', '--fp_16', action='store_true',
                        help='use fp 16', default=False)
    parser.add_argument('-wp', '--warm_up', default=CONFIG['wp'],
                        type=int, required=False)
    parser.add_argument('-dpt', '--dropout', default=CONFIG['dropout'],
                        type=float, required=False)
    parser.add_argument('-e_stop', '--eval_stop',
                        default=500, type=int, required=False)
    parser.add_argument('-bert_path', '--bert_path',
                        default=CONFIG['bert_path'], type=str, required=False)
    parser.add_argument('-data_path', '--data_path',
                        default=CONFIG['data_path'], type=str, required=False)
    parser.add_argument('-acc_step', '--accumulation_steps',
                        default=CONFIG['accumulation_steps'], type=int, required=False)
    parser.add_argument('-i', '--input', type=str,
                        help='load model', default='model.pkl')

    args = parser.parse_args()
    CONFIG['data_path'] = args.data_path
    CONFIG['lr'] = args.lr
    CONFIG['ptmlr'] = args.ptm_lr
    CONFIG['epochs'] = args.epochs
    CONFIG['bert_path'] = args.bert_path
    CONFIG['batch_size'] = args.batch
    CONFIG['dropout'] = args.dropout
    CONFIG['wp'] = args.warm_up
    CONFIG['p_unk'] = args.p_unk
    CONFIG['accumulation_steps'] = args.accumulation_steps
    CONFIG['task_name'] = args.task_name
    test_data_path = os.path.join(CONFIG['data_path'], 'kerc_test.csv')
    os.makedirs('vocabs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    seed = 1024
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    # torch.autograd.set_detect_anomaly(True)
    # model = PortraitModel(CONFIG)
    model = CRFModel(CONFIG)
    device = CONFIG['device']
    model.to(device)
    print('---config---')
    for k, v in CONFIG.items():
        print(k, '\t\t\t', v, flush=True)

    lst = os.listdir('./models')
    lst = list(filter(lambda item: item.endswith('.pkl'), lst))
    lst.sort(key=lambda x: os.path.getmtime(os.path.join('models', x)))
    model = torch.load(os.path.join('models', lst[-1]))
    print('checkpoint {} is loaded'.format(
        os.path.join('models', lst[-1])), flush=True)
    test(model, test_data_path)


# python train.py -tr -wp 0 -bsz 1 -acc_step 8 -lr 1e-4 -ptmlr 1e-5 -dpt 0.3 >> output.log 0.6505