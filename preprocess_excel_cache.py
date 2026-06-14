import argparse
import os

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from fieldroaddatapipeline.datareader import FieldRoadDataReader


def build_split_cache(path, output_file, max_len, drop_rate, num_workers):
    fileiter = FieldRoadDataReader(
        path,
        dataset_format='json',
        mode='Trace',
        num_workers=num_workers,
        max_len=max_len,
        drop_rate=drop_rate,
        scaler=MinMaxScaler(),
    )
    cached = []
    for cropped_points, trace_id, cropped_adjs, cropped_coordinates in fileiter:
        for points, adj, coordinates in zip(cropped_points, cropped_adjs, cropped_coordinates):
            rows, cols = np.nonzero(adj)
            cached.append(
                dict(
                    points=torch.from_numpy(points[:, :-1]).to(torch.float32),
                    labels=torch.from_numpy(points[:, -1].reshape(-1, 1)).to(torch.uint8),
                    edge_index=torch.from_numpy(np.stack([rows, cols]).astype('int32')),
                    trace_id=trace_id,
                    coordinates=torch.from_numpy(coordinates).to(torch.float32),
                )
            )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save(cached, output_file)
    return len(cached)


def main():
    parser = argparse.ArgumentParser(description='Preprocess Excel trajectory data into torch cache files.')
    parser.add_argument('--dataset-root', default='wheat')
    parser.add_argument('--split-dir', default='Non-Identically_Distributed_Coco')
    parser.add_argument('--trajectory-dir', default='sampled_wheat_43')
    parser.add_argument('--adj-dir', default='sampled_wheat_adj')
    parser.add_argument('--json-prefix', default='sampled_wheat_43')
    parser.add_argument('--out-dir', default='cache/wheat_non_iid')
    parser.add_argument('--max-len', type=int, default=1000)
    parser.add_argument('--drop-rate', type=float, default=0)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()

    split_names = ['train', 'valid', 'test']
    for split in split_names:
        path = dict(
            gnss=os.path.join(args.dataset_root, args.trajectory_dir),
            adj=os.path.join(args.dataset_root, args.adj_dir),
            json=os.path.join(args.dataset_root, args.split_dir, f'{args.json_prefix}_{split}.json'),
        )
        output_file = os.path.join(args.out_dir, f'{split}.pt')
        count = build_split_cache(path, output_file, args.max_len, args.drop_rate, args.num_workers)
        print(f'{split}: saved {count} samples to {output_file}', flush=True)


if __name__ == '__main__':
    main()
