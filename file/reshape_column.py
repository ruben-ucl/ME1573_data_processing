import argparse
import math
from pathlib import Path
import pandas as pd

CHUNK_SIZE = 5

def main():
    parser = argparse.ArgumentParser(description='Reshape a CSV column into rows of fixed width.')
    parser.add_argument('csv_path',   type=str, help='Path to input CSV file')
    parser.add_argument('col_index',  type=int, help='Zero-based column index to extract')
    args = parser.parse_args()

    src = Path(args.csv_path)
    df_in = pd.read_csv(src)

    col = df_in.iloc[:, args.col_index].tolist()

    n_rows = math.ceil(len(col) / CHUNK_SIZE)
    col_padded = col + [None] * (n_rows * CHUNK_SIZE - len(col))

    rows = [col_padded[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE] for i in range(n_rows)]
    df_out = pd.DataFrame(rows)

    out_path = src.with_stem(src.stem + '_reshaped')
    df_out.to_csv(out_path, index=False, header=False)
    print(f'Written {n_rows} rows to {out_path}')

if __name__ == '__main__':
    main()
