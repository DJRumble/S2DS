rmds
==============================

# Filter/Concatenate all the data into one file.

- 
1) Install the good package (anaconda, tqdm, seaborn)
On the command line : pip install tqdm seaborn
  ``` 
  pip install tqdm 
  pip install seaborn
  
  ```
If u are on the VPN, add : 

```
  pip install seaborn --proxy=thproxy.internet.point:8080
```

- 
2) Move up your raw data to the **raw folder**.
It should contain at least: 

s2ds1.txt
s2ds2.txt
s2ds3.txt
s2ds4.txt
s2ds1_update.txt

- 
3) In the command line, cd into the root of the directory.
Then, first export the ROOT_DIR environment variable which is
going to locate the path for this dir on your computer.

```
export ROOT_DIR=$PWD
```

Once, this is done, you can next generate the main data file
we are going use later for modelling.
``` 
 python src/data/gen_data.py
```

The cleaned data is now in the processed folder of the data folder. It
is called **processed_data.csv**. 
You can load it with pandas. 


