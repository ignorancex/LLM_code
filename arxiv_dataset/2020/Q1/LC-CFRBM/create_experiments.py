import json

out = {
    "experiments":
    [
    ]
}

for i in range(1,101):
    temp = {
        "name": "100k-u1xu1",
        "train_path": "ml-100k/u1.base",
        "test_path": "ml-100k/u1.test",
        "item_path": "ml-100k/u.item",
        "sep": "\t",
        "configs": [
            {
                "name": "nosparse_item_e100_b10_nh100_" + str(i),
                "epochs": 100,
                "batch_size": 10,
                "number_hidden": 100,
                "ks": [1],
                "momentums": [0.5, 0.6],
                "l_w": [0.0005],
                "l_v": [0.0005],
                "l_h": [0.0005],
                "decay": 0.0002
            }
        ]
    }
    out["experiments"].append(temp)


fout = open('nosparse_ibased.json','w')

json.dump(out, fout, indent = 4)
fout.close()