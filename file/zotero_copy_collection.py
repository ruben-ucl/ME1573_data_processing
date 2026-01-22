import csv
import argparse
import pathlib
import shutil
import os

parser = argparse.ArgumentParser(description='Copy PDFs from Zotero to the given destination.')
parser.add_argument('-c', '--csv', type=pathlib.Path,
              required=True, help='CSV file exported from Zotero')
parser.add_argument('-d', '--dest', type=pathlib.Path,
              default='./', help='Destination folder for the PDFs')
args = parser.parse_args()

copySuccess = 0
copyFail = []

output = args.dest.absolute()

if not os.path.exists(output):
    os.makedirs(output)
    
print('Copying collection PDFs to... {}'.format(output))

with open(args.csv.absolute(), newline='', encoding='utf-8-sig') as csvfile:
    cr = csv.DictReader(csvfile)
    for row in cr:
      files = row["File Attachments"].split('; ')
      for fp in files:
          print("Copying... {}".format(fp))
          try:
              shutil.copy(fp, output)
              copySuccess = copySuccess+1
          except Exception as e:
            copyFail.append(fp)
            # print(e)
        
print("Done. {} Succeed, {} Failed.".format(copySuccess, len(copyFail)))
if len(copyFail) > 0:
    print('Failed files:')
    for fp in copyFail:
        if fp == '':
            print('[Missing filepath]')
        else:
            print(fp)
