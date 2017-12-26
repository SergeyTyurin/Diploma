#!/bin/bash

if [[ $# == 0 || $# == 1 || $# == 2  ]]; then
    echo "Требуется 3 армумента: 1 - модель, 2 - директория с логом, 3 - имя файла"
    exit 0
else
    if ! [[ -d $1 ]]; then
		echo "Модель $1 не найдена"
		exit 0
	elif ! [[ -d $1/$2 ]]; then
		echo "Директория $1/$2 не найдена"
		exit 0
	else
		if ! [[ -f $1/$2/$3 ]]; then
			echo "Файл $1/$2/$3 не найден"
			exit 0
		fi
	fi
fi

if ! [[ -d $1/$2/data ]]; then
	mkdir $1/$2/data
else
	echo $1/$2/data
fi

if ! [[ -d $1/$2/data/train ]]; then
	mkdir $1/$2/data/train
else
	echo $1/$2/data/train
fi

if ! [[ -d $1/$2/data/val ]]; then
	mkdir $1/$2/data/val
else
	echo $1/$2/data/val
fi

TRAIN="$1/$2/data/train"
VAL="$1/$2/data/val"

grep "^.*Iteration.*loss =.*$" $1/$2/$3 | grep -o "Iteration [0-9]*" | grep -o [0-9][0-9]* > $TRAIN/Train_iterations.txt
grep "^.*Train.*accuracy =.*$" $1/$2/$3 | grep -o "accuracy = [0-9]*.*$" | grep -o "[0-9]*[.]*[0-9]*$" > $TRAIN/Train_accuracy.txt
grep "^.*Train.*loss =.*$" $1/$2/$3 | grep -o "loss = [0-9]*\.[0-9]*.*" | grep -o "[0-9]*\.[0-9]*" > $TRAIN/Train_loss.txt
grep "Test" $1/$2/$3 | grep -o "Iteration \d*" | grep -o "\d\d*" > $VAL/Val_iterations.txt
grep "Test" $1/$2/$3 | grep -o "loss = [0-9]\.[0-9]*.*" | grep -o "[0-9]\.[0-9]*" > $VAL/Val_loss.txt
grep "Test" $1/$2/$3 | grep -o "accuracy = [0-9]\.[0-9]*.*" | grep -o "[0-9]\.[0-9]*" > $VAL/Val_acc.txt

~/Diploma/env/bin/python Create_CSV.py --train_csv_file $TRAIN/train.csv\
                                       --val_csv_file $VAL/val.csv\
                                       --train_iterations_file $TRAIN/Train_iterations.txt\
                                       --train_accuracy_file $TRAIN/Train_accuracy.txt\
                                       --train_loss_file $TRAIN/Train_loss.txt\
                                       --val_iterations_file $VAL/Val_iterations.txt\
                                       --val_accuracy_file $VAL/Val_acc.txt\
                                       --val_loss_file $VAL/Val_loss.txt\

rm $TRAIN/Train_iterations.txt
rm $TRAIN/Train_accuracy.txt
rm $TRAIN/Train_loss.txt
rm $VAL/Val_iterations.txt
rm $VAL/Val_loss.txt
rm $VAL/Val_acc.txt
