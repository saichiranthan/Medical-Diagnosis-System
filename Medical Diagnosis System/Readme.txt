Team 12 Project INSTRUCTIONS

1.Install the Required Python modules :

sklearn
pandas
numpy
tkinter
customtkinter
matplotlib
graphviz

2. Change the location of datasets
Note: Datasets given in the datasets folder in the drive

df=pd.read_csv("D:\\Btech_AI\\3rdsem\\DMS\\codes\\project\\part2\\Prototype.csv")
tr=pd.read_csv("D:\\Btech_AI\\3rdsem\\DMS\\codes\\project\\part2\\Prototype-1.csv")

Here training data is readby 'df' and testing data is read by 'tf'
Do change the path as per your dataset location

3.Change the location of saving graph image to your system location
graph.render("D:\\Btech_AI\\3rdsem\\DMS\\codes\\project\\part2\\decision_tree_graphivz")

and also the image opening path should be same path where you save it
img_path = "D:\\Btech_AI\\3rdsem\\DMS\\codes\\project\\part2\\decision_tree_graphivz.png"

4.NOTE similar changes to be done for comparison model also i.e RandomForest model 