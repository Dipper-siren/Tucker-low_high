from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
from collections import Counter
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse

    
class Experiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200, 
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False, 
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0., loss_weight=2., regularization=1.0, bias = 0.2):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.loss_weight = loss_weight
        self.cuda = cuda
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2, "bias": bias}
        self.regularization = regularization
        self.max_hit10 = 0
        self.max_hit3 = 0
        self.max_hit1 = 0
        self.max_mrr = 0
        self.bias = bias
    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def get_two_rela_idxs(self, data_idxs):
        two_rela_idxs = []
        # flit_idxs=[]
        rt_vocab = self.get_rt_vocab(data_idxs)
        for i in range(len(data_idxs)):
            result_rt = rt_vocab[data_idxs[i][2]]
            for r in range(len(result_rt)):
                two_rela_idxs.append((data_idxs[i][0], (data_idxs[i][1]+1) * len(d.relations) + result_rt[r][0], result_rt[r][1]))
                # print(data_idxs[i][1],result_rt[r][0]
                #     ,(data_idxs[i][1]+1) * len(d.relations) + result_rt[r][0])
        return two_rela_idxs

    def get_rt_vocab(self, data):
        rt_vocab = defaultdict(list)   # 当查询对象不在字典中时，返回一个空表项。
        for triple in data:
            rt_vocab[triple[0]].append((triple[1], triple[2]))   # 建立根据头查询关系和尾实体的字典
        #  print(er_vocab)
        return rt_vocab
    
    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))  # 为batch_size*实体个数的矩阵。
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.  # 对应批次的尾实体的序号被标记为1为真，其余为假.
        targets = torch.FloatTensor(targets)  # 将nparray转化为tensor
        if self.cuda:
            targets = targets.cuda()  # 将targets转移到GPU上。
        return np.array(batch), targets

    
    def evaluate(self, model, data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))
        
        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:,0])
            r_idx = torch.tensor(data_batch[:,1])
            e2_idx = torch.tensor(data_batch[:,2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions_1, predictions_2, margin_loss = model.forward(e1_idx, r_idx)
            predictions = (predictions_1 + predictions_2) / 2

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j,e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j]==e2_idx[j].item())[0][0]
                ranks.append(rank+1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))
        return np.mean(hits[9]), np.mean(hits[2]), np.mean(hits[0]), np.mean(1./np.array(ranks))




    def train_and_eval(self):
        print("Training the TuckER model...")
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        # train_two_idxs = self.get_two_rela_idxs(train_data_idxs)
        # print(len(train_two_idxs))
        # # train_two_idxs = list(set(train_two_idxs))  # 去除二步关系中重复的部分。
        # count = dict(Counter(train_two_idxs))  # 保留二步关系中重复的部分。
        # train_two_idxs = [key for key, value in count.items() if value > 1]
        # # 是索引集合。（2，2，44）数字代表实体和关系在字典中的索引。

        model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        if self.cuda:
            model.cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        # er_vocab_two = self.get_er_vocab(train_two_idxs)
        # er_vocab_pairs_2 = list(er_vocab_two.keys())  # 获取查询词典的关键词部分

        print("Starting training...")
        for it in range(1, self.num_iterations+1):
            start_train = time.time()
            model.train()    
            losses = []

            np.random.shuffle(train_data_idxs)
            er_vocab = self.get_er_vocab(train_data_idxs)
            er_vocab_pairs = list(er_vocab.keys())

            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:,0])
                r_idx = torch.tensor(data_batch[:,1])  
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                predictions_1, predictions_2, margin_loss = model.forward(e1_idx, r_idx)
                if self.label_smoothing:
                    targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))
                # if self.regularization != 0.0:
                #     # Use L3 regularization for Tucker  # 做L3正则化。
                #     regular = self.regularization * (
                #             model.E.norm(p=3) ** 3 +
                #             model.R.norm(p=3).norm(p=3) ** 3
                #     )
                # DURA 正则项

                # triples = train_data_idxs[j:j+self.batch_size]
                # print(triples)
                # h = [x[0] for x in triples]
                # h = torch.tensor(h)
                # h = h.cuda()
                # r = [x[1] for x in triples]
                # r = torch.tensor(r)
                # r = r.cuda()
                # t = [x[2] for x in triples]
                # t = torch.tensor(t)
                # t = t.cuda()
                loss = model.loss(predictions_1, targets) + model.loss(predictions_2, targets) + self.loss_weight * margin_loss
                loss.backward()
                opt.step()
                for i in model.W_low:
                    i.data.clamp_(0, 1)   # 控制W_low的取值范围。
                for i in model.W_high:
                    i.data.clamp_(0, 1)   # 控制W_low的取值范围。
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            print("iter:%d" % it)
            print(time.time()-start_train)
            # print("bias: %f " % model.bias)
            print("loss: %f" % np.mean(losses))
            model.eval()
            with torch.no_grad():
                print("Validation:")
                self.evaluate(model, d.valid_data)
                if not it%2:
                    print("Test:")
                    start_test = time.time()
                    h10,h3,h1,mrr = self.evaluate(model, d.test_data)
                    print(time.time()-start_test)
                    if h10 > self.max_hit10:
                        self.max_hit10 = h10
                    if h3 > self.max_hit3:
                        self.max_hit3 = h3
                    if h1 > self.max_hit1:
                        self.max_hit1 = h1
                    if mrr > self.max_mrr:
                        self.max_mrr = mrr
                    print('max:', self.max_hit10, self.max_hit3, self.max_hit1, self.max_mrr)


        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15K-237", nargs="?",
                    help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--num_iterations", type=int, default=800, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                    help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--dr", type=float, default=1, nargs="?",
                    help="Decay rate.")
    parser.add_argument("--edim", type=int, default=200, nargs="?",
                    help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=200, nargs="?",
                    help="Relation embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?",
                    help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?",
                    help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?",
                    help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                    help="Amount of label smoothing.")
    parser.add_argument("--loss_weight", type=float, default=2., nargs="?",
                        help="control the two-rela-loss.")
    parser.add_argument('--regularization', type=float, default=1.0)
    parser.add_argument('--bias', type=float, default=0.2)

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True 
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed) 
    d = Data(data_dir=data_dir, reverse=True)
    experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr, 
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                            input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1, 
                            hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing)
    experiment.train_and_eval()
                

