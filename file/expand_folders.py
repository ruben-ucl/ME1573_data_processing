import glob, os
from pathlib import Path


target_path = Path('J:\AMPM')
for folder in glob.glob(f'{target_path}/*/'):
    trackid = folder[8:12]
    print(f'\n{trackid}')
    for file in glob.glob(f'{folder}*100K*.dat'):
        print('moving ', file, ' to ', Path(target_path, f'{trackid}_{Path(file).name}'))
        os.rename(file, Path(target_path, f'{trackid}_{Path(file).name}'))