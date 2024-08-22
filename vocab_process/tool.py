import pickle
import argparse

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--corpus_path", type=str, required=True,
                        help="Path of the corpus to count instances.")
    
    args = parser.parse_args()

    dataset_path = args.corpus_path

    dataset_reader = open(dataset_path, "rb")

    count = 0
    while True:
        try:
            instance = pickle.load(dataset_reader)
            count += 1
        except EOFError:
            break

    print("Number of Python objects:", count)

if __name__ == "__main__":
    main()