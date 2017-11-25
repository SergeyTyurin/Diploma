import csv
import os
import argparse

file_path = os.path.dirname(__file__)

def create_train_csv(csv_file, iterations_file, loss_file):
    try:
        train_csvfile = open(os.path.join(file_path,csv_file),'w')
        train_iters = open(os.path.join(file_path,iterations_file), 'r').readlines()
        train_loss = open(os.path.join(file_path,loss_file), 'r').readlines()

        writer = csv.writer(train_csvfile)
        writer.writerow(['Iteration', 'Loss'])
        for it, l in zip(train_iters, train_loss):
            writer.writerow([int(it.strip('\n')),float(l.strip('\n'))])

    except Exception as e:
        print e

def create_val_csv(csv_file, iterations_file, acc1_file, acc2_file, acc3_file):
    try:
        val_csvfile = open(os.path.join(file_path,csv_file),'w')
        val_iters = open(os.path.join(file_path,iterations_file),'r')
        val_acc1 = open(os.path.join(file_path,acc1_file),'r')
        val_acc2 = open(os.path.join(file_path,acc2_file),'r')
        val_acc3 = open(os.path.join(file_path,acc3_file),'r')

        writer_val = csv.writer(val_csvfile)
        writer_val.writerow(["Iteration","Accuracy1","Accuracy2","Accuracy3"])
        for it,acc1,acc2,acc3 in zip(val_iters,val_acc1,val_acc2,val_acc3):
            writer_val.writerow([int(it.strip('\n')),
                                 float(acc1.strip('\n')),
                                 float(acc2.strip('\n')),
                                 float(acc3.strip('\n'))])

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

    parser.add_argument("--train_loss_file", type=str, default=None,
                        required=True)

    parser.add_argument("--val_iterations_file", type=str, default=None,
                        required=True)

    parser.add_argument("--val_accuracy1_file", type=str, default=None,
                        required=True)

    parser.add_argument("--val_accuracy2_file", type=str, default=None,
                        required=True)

    parser.add_argument("--val_accuracy3_file", type=str, default=None,
                        required=True)

    args = parser.parse_args()

    create_train_csv(args.train_csv_file,
                     args.train_iterations_file,
                     args.train_loss_file
                     )

    create_val_csv(args.val_csv_file,
                   args.val_iterations_file,
                   args.val_accuracy1_file,
                   args.val_accuracy2_file,
                   args.val_accuracy3_file,
                   )

if __name__=="__main__":
    main()
