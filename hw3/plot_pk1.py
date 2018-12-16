import pickle


def setup_inputs():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    parser.add_argument('--method', type=str, default='beat_mean')
    return parser.parse_args()


def main():
    args = setup_inputs()
    with open(args.filename, 'rb') as f:
        data = pickle.load(f)
        print(data)


if __name__ == '__main__':
    main()
