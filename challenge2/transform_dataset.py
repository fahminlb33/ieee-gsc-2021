import csv
import pandas as pd
import argparse

# trainng arguments
parser = argparse.ArgumentParser(description="Perform dataset transformation to train a model")
parser.add_argument("train_dataset",
                    type=str,
                    help="Path to dataset file used to train the model")
parser.add_argument("output_name",
                    type=str,
                    help="Path to output file")
parser.add_argument("--mode",
                    type=str,
                    choices=["sparknlp", "tensorflow"],
                    default="sparknlp",
                    help="Transform the dataset suitable to use with sparknlp or tensorflow")

# parse arguments
args = vars(parser.parse_args())

class_map = {
    0: "optimistic",
    1: "thankful",
    2: "empathetic",
    3: "pessimistic",
    4: "anxious",
    5: "sad",
    6: "annoyed",
    7: "denial",
    8: "surprise",
    9: "official_report",
    10: "joking"
}

def process_sparknlp(row):
    if row[0] == "ID":
        return None

    current_classes = []
    classes = list(map(int, row[2].split(" ")))
    for i in range(0, 11):
        if i in classes:
            current_classes.append(class_map[i])

    return (row[0], row[1], current_classes) # target class as list

def process_tensorflow(row):
    if row[0] == "ID":
        return None

    current_row = [row[0], row[1]]
    classes = list(map(int, row[2].split(" ")))
    for i in range(0, 11):
        if i in classes:
            current_row.append(1)
        else:
            current_row.append(0)

    return tuple(current_row) # one hot encoding

# open the file
with open(args["train_dataset"]) as csv_file:
    # open CSV reader
    csv_reader = csv.reader(csv_file, delimiter=',')
    df = None

    # perform transformation for each row
    if args["mode"] == "sparknlp":
        records = [process_sparknlp(row) for row in csv_reader][1:]
        df = pd.DataFrame.from_records(records, columns=["id", "text", "labels"])
    else:
        records = [process_tensorflow(row) for row in csv_reader][1:]
        df = pd.DataFrame.from_records(records, columns=["ID", "Tweet"] + [x for x in range(0, 11)])

    print(f'Processed {len(df)} lines.')

    # show top 5 data
    print(df.head())

    # save to file
    if args["mode"] == "sparknlp":
        df.to_parquet(args["output_name"], compression="gzip")
    else:
        df.to_csv(args["output_name"], index=None)