# NaiveBayesianClassifier
Implementation of Naive Bayesian Classifier for Census-Income data using Python

## File Structure:
Following are the files of this project.
- **csc869proj1.py**	-	This file contains the code for naive-baysian and for reading the data file
- **adult.txt**		    -	This is the data file. This file needs to be in the same directory 
					              as the csc869proj1.py file for the code to execute successfully.
					
Please Note:
`Adult.csv` has be modified to `adult.txt` so that Tableau can easily read it.
Also the column names have been manually added to the first line of the data file , before executing any code.

### Dataset:
Original dataset has been downloaded from the following link: http://archive.ics.uci.edu/ml/datasets/Adult

### Libraries Used:
Following libraries have been used:
`pandas, numpy`	-	  Used the dataframes and other datastructures to read and parse data in a better way.
`deepcopy`		  -	  To make multiple deepcopies of create data structure during the code.
`time`			    -	  To calculate the execution time of the program.

### Data Imputation:
For handling missing values following methods have been used:
1)	Replace missing values using `mean/mode`
	Following lines needs to be uncommented
	
	  line 292:	#Method 1 for handling missing values: Replace with mean/mode for continuous/discrete columns
    line 293:	replace_missing_nonnumeric_values_with_mode(data)
	
2)	Remove the records containing missing values

	  Comment the code on line 292 (abv method) and  uncomment line 295
	
	  line 295:	#Method 2 for handling missing values: Remove records with null value in any column
    line 296:	#data = remove_record_with_missing_value(data)
	
### Discretization:
For handling continuous variables following two methods have been used:
1)	Equi-Width Binning
	To activate this method uncomment the following lines of code and comment the part of gaussin method
	line 134:	#classes = classes_using_equi_width(data)
    line 135:	#flag = True
	
2)	Gaussian Distribution
	To activate this method uncomment the following lines of code and comment the part of equi-width
	line 136:	classes = classes_using_gaussian(data)
	line 137:	flag = False
