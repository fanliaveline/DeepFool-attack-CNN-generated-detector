import os
import csv
import torch

from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
from eval_config import *


# Running tests
opt = TestOptions().parse(print_options=False)
model_name = os.path.basename(model_path).replace('.pth', '')
rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision','rcc','fcc','auc']]

print("{} model testing on...".format(model_name))
for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)
    opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    #print(opt.classes)
    opt.no_resize = True    # testing without resizing by default

    model = resnet50(num_classes=1)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, ap, rcc, fcc,auc, _, _ = validate(model, opt)
    rows.append([val, acc, ap,rcc,fcc,auc])
    print("({}) acc: {}; ap: {};rcc: {}；fcc:{};auc：{}".format(val, acc, ap, rcc, fcc,auc))

csv_name = results_dir + '/{}.csv'.format(model_name)
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)
