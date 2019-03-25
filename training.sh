
for j in `seq 1 2000`;
do
	 python3 training_new.py
	 for i in `seq 1 10`;
	 do
	     python3 main.py
	 done
    
	 python3 training.py
	 for i in `seq 1 10`;
	 do
	     python3 main.py
	 done

done
