import csv
import os
import argparse

file_path = os.path.dirname(__file__)


def create_train_csv(csv_file, iterations_file, acc_file, loss_file):
    try:
        train_csvfile = open(os.path.join(file_path,csv_file),'w')
        train_iters = open(os.path.join(file_path,iterations_file), 'r').readlines()
        train_loss = open(os.path.join(file_path,loss_file), 'r').readlines()
        train_acc = open(os.path.join(file_path,acc_file), 'r').readlines()

        writer = csv.writer(train_csvfile)
        writer.writerow(['Iteration', 'Loss', 'Accuracy'])
        for it, l, acc in zip(train_iters, train_loss,train_acc):
            writer.writerow([int(it.strip('\n')),float(l.strip('\n')),float(acc.strip('\n'))])

    except Exception as e:
        print e

def create_val_csv(csv_file, iterations_file, acc_file ,loss_file):
    try:
        val_csvfile = open(os.path.join(file_path,csv_file),'w')
        val_iters = open(os.path.join(file_path,iterations_file),'r').readlines()
        val_loss = open(os.path.join(file_path,loss_file),'r').readlines()
        val_acc = open(os.path.join(file_path,acc_file),'r').readlines()

        writer_val = csv.writer(val_csvfile)
        writer_val.writerow(["Iteration","Loss","Accuracy"])
        for it,l,acc in zip(val_iters,val_loss,val_acc):
            writer_val.writerow([int(it.strip('\n')),float(l.strip('\n')),float(acc.strip('\n'))])

    except Exception as e:
        print e

def main():
    parser = argparse.ArgumentParser(
        description="Create CSV"
    )

    parser.add_argument("--train_csv_file", type=str, default=None,
                        required=True)

    parser.add_argument("--val_csv_file", type=str, default=None,
                        required=True)

    parser.add_argument("--train_iterations_file", type=str, default=None,
                        required=True)

    parser.add_argument("--train_accuracy_file", type=str, default=None,
                        required=True)

    parser.add_argument("--train_loss_file", type=str, default=None,
                        required=True)

    parser.add_argument("--val_iterations_file", type=str, default=None,
                        required=True)

    parser.add_argument("--val_accuracy_file", type=str, default=None,
                        required=True)

    parser.add_argument("--val_loss_file", type=str, default=None,
                        required=True)

    args = parser.parse_args()

    create_train_csv(args.train_csv_file,
                     args.train_iterations_file,
                     args.train_accuracy_file,
                     args.train_loss_file
                     )

    create_val_csv(args.val_csv_file,
                   args.val_iterations_file,
                   args.val_accuracy_file,
                   args.val_loss_file
                   )

if __name__=="__main__":
    main()
