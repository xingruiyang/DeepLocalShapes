import json
import argparse
import os

shapenet_id = {'plane': }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    args = parser.parse_args()

    with open(os.path.join(args.dataset, 'taxonomy.json')) as f:
        data = json.load(f)
    print(data)